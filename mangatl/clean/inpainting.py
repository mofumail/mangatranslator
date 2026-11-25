import math

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from mangatl.common.caching import get_cache
from mangatl.common.device import get_best_device, get_best_dtype
from mangatl.ml.model_manager import get_model_manager
from mangatl.common.cancellation import raise_if_cancelled
from mangatl.common.log import log_message

# Dynamic edge feathering scaled to bubble size; clamped to prevent sharp seams on small text or bleed on large text
BLUR_SCALE_FACTOR = 0.1
MIN_BLUR_RADIUS = 1
MAX_BLUR_RADIUS = 10


# Surrounding context ratio and pixel ceiling to ensure clean blending without diluting inpaint resolution
CONTEXT_PADDING_RATIO = 0.5
MAX_CONTEXT_PADDING = 80


def _prompt_value_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(_prompt_value_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_prompt_value_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _prompt_value_to_cpu(item) for key, item in value.items()}
    return value


def _prompt_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_prompt_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_prompt_value_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _prompt_value_to_device(item, device) for key, item in value.items()
        }
    return value


def make_inpainter(
    method: str,
    *,
    device: torch.device | None,
    luminance_correction: bool,
    verbose: bool = False,
):
    """The inpainting model for a method key: LaMa Manga ("lama_manga"), else None."""
    if method == "lama_manga":
        return LamaInpainter(
            device=device, luminance_correction=luminance_correction, verbose=verbose
        )
    return None


def _lookup_inpaint_patch(
    cache,
    image_cropped_pil: Image.Image,
    mask_crop_np: np.ndarray,
    seed: int,
    num_inference_steps: int,
    residual_diff_threshold: float,
    guidance_scale: float,
    prompt: str,
    cache_params: dict,
) -> tuple[str | None, Image.Image | None]:
    """Return (cache key, cached patch); (None, None) when the seed disables caching."""
    if not cache.should_use_inpaint_cache(seed):
        return None, None

    # Downsample mask signature to reduce sensitivity to minor jitter
    if mask_crop_np.size > 0:
        sig_h = min(64, max(4, mask_crop_np.shape[0]))
        sig_w = min(64, max(4, mask_crop_np.shape[1]))
        mask_sig = (
            torch.from_numpy(mask_crop_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )
        mask_sig = torch.nn.functional.interpolate(
            mask_sig, size=(sig_h, sig_w), mode="bilinear", align_corners=False
        )
        mask_sig_np = (mask_sig > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
    else:
        mask_sig_np = mask_crop_np

    cache_key = cache.get_inpaint_cache_key(
        image_cropped_pil,
        mask_sig_np,
        seed,
        num_inference_steps,
        residual_diff_threshold,
        guidance_scale,
        prompt,
        cache_params,
    )
    cached_patch = cache.get_inpainted_image(cache_key)
    if cached_patch is not None:
        log_message("  - Using cached inpainting patch", always_print=True)
    return cache_key, cached_patch


def _clip_composite_mask(
    mask_for_composite: torch.Tensor,
    composite_clip_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    """Zero the crop's composite mask outside `composite_clip_bbox` (image coordinates).

    `crop_box` is (x, y, width, height) of the crop within the image.
    """
    img_w, img_h = image_size
    x, y, width, height = crop_box
    clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox
    clip_x1 = max(0, min(img_w, clip_x1))
    clip_x2 = max(0, min(img_w, clip_x2))
    clip_y1 = max(0, min(img_h, clip_y1))
    clip_y2 = max(0, min(img_h, clip_y2))

    start_x = max(0, clip_x1 - x)
    end_x = min(width, clip_x2 - x)
    start_y = max(0, clip_y1 - y)
    end_y = min(height, clip_y2 - y)

    clipped_mask = torch.zeros_like(mask_for_composite)
    if end_x > start_x and end_y > start_y:
        clipped_mask[:, start_y:end_y, start_x:end_x] = mask_for_composite[
            :, start_y:end_y, start_x:end_x
        ]
    return clipped_mask


def _feathered_crop_mask(mask_crop_np: np.ndarray, blur_radius: int) -> torch.Tensor:
    """Composite alpha for a crop: 1 inside the mask, fading to 0 over `blur_radius` px."""
    if blur_radius <= 0:
        return torch.from_numpy(mask_crop_np.astype(np.float32))[None, ...]
    d_out = distance_transform_edt(~mask_crop_np)
    d_in = distance_transform_edt(mask_crop_np)
    alpha = np.zeros_like(d_out, np.float32)
    alpha[d_in > 0] = 1.0
    ramp = np.clip(1.0 - (d_out / blur_radius), 0.0, 1.0)
    alpha[d_out > 0] = ramp[d_out > 0]
    return torch.from_numpy(alpha)[None, ...]


def _paste_patch(
    image_pil: Image.Image,
    patch_pil: Image.Image,
    mask_for_composite: torch.Tensor,
    x1: int,
    y1: int,
) -> Image.Image:
    """Blend `patch_pil` into `image_pil` at (x1, y1) through the composite mask."""
    dest_tensor = torch.from_numpy(
        np.asarray(image_pil, dtype=np.float32) / 255.0
    ).unsqueeze(0)
    src_tensor = torch.from_numpy(
        np.asarray(patch_pil, dtype=np.float32) / 255.0
    ).unsqueeze(0)

    # Fix channel mismatch (e.g. RGBA destination vs RGB source)
    dest_channels = dest_tensor.shape[-1]
    src_channels = src_tensor.shape[-1]

    if dest_channels > src_channels:
        alpha_fill = torch.ones(
            (*src_tensor.shape[:-1], dest_channels - src_channels),
            device=src_tensor.device,
            dtype=src_tensor.dtype,
        )
        src_tensor = torch.cat([src_tensor, alpha_fill], dim=-1)
    elif src_channels > dest_channels:
        src_tensor = src_tensor[..., :dest_channels]

    dest_tensor = dest_tensor.movedim(-1, 1)
    src_tensor = src_tensor.movedim(-1, 1)

    mask_interp = torch.nn.functional.interpolate(
        mask_for_composite.reshape(
            (-1, 1, mask_for_composite.shape[-2], mask_for_composite.shape[-1])
        ),
        size=(src_tensor.shape[2], src_tensor.shape[3]),
        mode="bilinear",
    )

    composited = dest_tensor.clone()
    visible_h = min(src_tensor.shape[2], dest_tensor.shape[2] - y1)
    visible_w = min(src_tensor.shape[3], dest_tensor.shape[3] - x1)

    if visible_h > 0 and visible_w > 0:
        src_portion = src_tensor[:, :, :visible_h, :visible_w]
        mask_portion = mask_interp[:, :, :visible_h, :visible_w]
        dest_portion = composited[:, :, y1 : y1 + visible_h, x1 : x1 + visible_w]

        blended = src_portion * mask_portion + dest_portion * (1 - mask_portion)
        composited[:, :, y1 : y1 + visible_h, x1 : x1 + visible_w] = blended

    composited = composited.movedim(1, -1)
    return Image.fromarray((composited[0].cpu().numpy() * 255).astype("uint8"))


def _match_luminance(
    generated_pil: Image.Image,
    original_crop_pil: Image.Image,
    mask_crop_np: np.ndarray,
    verbose: bool = False,
    boundary_width: int = 12,
) -> Image.Image:
    """Match patch luminance and color to surrounding context.

    Uses a local boundary band around the mask to avoid distant margin/gutter
    contamination, applies an additive luminance shift to preserve internal
    contrast without false scaling artifacts, and forces true neutral chroma
    on monochrome manga to eliminate diffusion color casts.

    Args:
        generated_pil: Generated patch (at crop size)
        original_crop_pil: Original cropped region
        mask_crop_np: Boolean mask of inpaint region within crop
        verbose: Whether to print verbose output
        boundary_width: Width in pixels of the local reference boundary ring

    Returns:
        Luminance-corrected generated patch
    """
    mask_bool = np.asarray(mask_crop_np, dtype=bool)
    if not np.any(mask_bool):
        return generated_pil

    # Prefer immediate boundary ring to avoid distant gutters/borders skewing statistics
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (boundary_width * 2 + 1, boundary_width * 2 + 1)
    )
    dilated_mask = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0
    boundary_band = dilated_mask & (~mask_bool)

    ref_mask = boundary_band if np.count_nonzero(boundary_band) >= 32 else (~mask_bool)
    if not np.any(ref_mask):
        return generated_pil

    if original_crop_pil.mode != "RGB":
        original_crop_pil = original_crop_pil.convert("RGB")
    if generated_pil.mode != "RGB":
        generated_pil = generated_pil.convert("RGB")

    original_np = np.asarray(original_crop_pil)
    generated_np = np.asarray(generated_pil).copy()

    orig_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    orig_l_mean = float(np.mean(orig_lab[:, :, 0][ref_mask]))
    gen_l_mean = float(np.mean(gen_lab[:, :, 0][ref_mask]))
    l_shift = orig_l_mean - gen_l_mean

    if abs(l_shift) >= 1.0:
        log_message(
            f"  - Luminance correction: L shift {l_shift:+.1f} (mean {gen_l_mean:.1f}->{orig_l_mean:.1f})",
            verbose=verbose,
        )
        gen_lab[:, :, 0][mask_bool] = np.clip(
            gen_lab[:, :, 0][mask_bool] + l_shift, 0, 255
        )

    # In OpenCV LAB, neutral gray is a=128, b=128
    a_dev = float(np.mean(np.abs(orig_lab[:, :, 1][ref_mask] - 128.0)))
    b_dev = float(np.mean(np.abs(orig_lab[:, :, 2][ref_mask] - 128.0)))
    is_monochrome = (a_dev < 3.0) and (b_dev < 3.0)

    if is_monochrome:
        gen_lab[:, :, 1][mask_bool] = 128.0
        gen_lab[:, :, 2][mask_bool] = 128.0
    else:
        for ch in (1, 2):
            orig_ch_mean = float(np.mean(orig_lab[:, :, ch][ref_mask]))
            gen_ch_mean = float(np.mean(gen_lab[:, :, ch][ref_mask]))
            ch_shift = orig_ch_mean - gen_ch_mean
            if abs(ch_shift) > 1.0:
                gen_lab[:, :, ch][mask_bool] = np.clip(
                    gen_lab[:, :, ch][mask_bool] + ch_shift, 0, 255
                )

    corrected_np = cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(corrected_np)


class _CropInpainter:
    """Inpaints a padded crop around the mask and blends the patch back in.

    Subclasses set `model_key` and `label` and provide load_models, unload_models,
    and _generate_patch.
    """

    model_key = ""
    label = ""
    PROMPT = ""
    GUIDANCE_SCALE = 1.0
    # Diffusion models need the VRAM the text segmentation model holds
    PARKS_TEXT_SAM = True

    # DiT patch requirements (multiples of 16) and VRAM cap (~4MP max inference)
    MIN_RESOLUTION = 64
    MAX_RESOLUTION = 2048
    RESOLUTION_MULTIPLE = 16
    MAX_INFERENCE_PIXELS = 4_000_000
    PADDING_MULTIPLIER = (
        2.0  # Wider context helps preserve surrounding line art and details
    )

    def __init__(
        self,
        *,
        backend: str,
        device: torch.device | None,
        num_inference_steps: int,
        luminance_correction: bool,
        upscale_small_crops: bool,
        verbose: bool,
    ):
        self.backend = backend
        self.num_inference_steps = num_inference_steps
        self.luminance_correction = luminance_correction
        self.upscale_small_crops = upscale_small_crops
        self.verbose = verbose

        self.DEVICE = device if device is not None else get_best_device()
        self.DTYPE = get_best_dtype(self.DEVICE)
        self.manager = get_model_manager()
        self.cache = get_cache()
        self.pipeline = None

    def load_models(self):
        raise NotImplementedError

    def unload_models(self):
        raise NotImplementedError

    def _generate_patch(
        self,
        image: Image.Image,
        mask: np.ndarray,
        width: int,
        height: int,
        seed: int,
        verbose: bool,
    ) -> Image.Image:
        """Inpaint the scaled crop; `mask` is the crop's mask at its original size."""
        raise NotImplementedError

    def _snap_dimension(self, dim: int) -> int:
        """Round dimension down to a multiple of RESOLUTION_MULTIPLE, at least MIN_RESOLUTION."""
        dim = max(self.MIN_RESOLUTION, dim)
        return (dim // self.RESOLUTION_MULTIPLE) * self.RESOLUTION_MULTIPLE

    def _quantize_dimension(self, dim: int) -> int:
        """Quantize dimension to be a multiple of RESOLUTION_MULTIPLE within allowed range."""
        return self._snap_dimension(min(self.MAX_RESOLUTION, dim))

    def _expand_bounds_to_min_size(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        target_w = min(self.MIN_RESOLUTION, img_w)
        target_h = min(self.MIN_RESOLUTION, img_h)

        width = x2 - x1
        if width < target_w:
            extra = target_w - width
            x1 = max(0, x1 - extra // 2)
            x2 = min(img_w, x2 + extra - extra // 2)
            if x2 - x1 < target_w:
                if x1 == 0:
                    x2 = min(img_w, target_w)
                else:
                    x1 = max(0, img_w - target_w)

        height = y2 - y1
        if height < target_h:
            extra = target_h - height
            y1 = max(0, y1 - extra // 2)
            y2 = min(img_h, y2 + extra - extra // 2)
            if y2 - y1 < target_h:
                if y1 == 0:
                    y2 = min(img_h, target_h)
                else:
                    y1 = max(0, img_h - target_h)

        return x1, y1, x2, y2

    def _compute_luminance_stats(
        self, image_np: np.ndarray, mask_np: np.ndarray
    ) -> tuple[float, float]:
        """Compute mean and std of luminance (LAB L channel) for masked pixels.

        Args:
            image_np: RGB image array (H, W, 3), uint8
            mask_np: Boolean mask where True = pixels to include

        Returns:
            (mean, std) of L channel values (0-255 scale, cv2 LAB)
        """
        if not np.any(mask_np):
            return 127.5, 30.0  # Neutral fallback

        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_values = lab[:, :, 0][mask_np].astype(np.float32)

        return float(np.mean(l_values)), float(np.std(l_values)) + 1e-6

    def _prepare_image_for_inference(
        self,
        image_pil: Image.Image,
        verbose: bool = False,
    ) -> tuple[Image.Image, int, int]:
        """Prepare image for inference.

        Optionally scales the image to approximately 1 megapixel while maintaining
        aspect ratio. When that scaling is disabled, large crops are still capped
        to 4MP. Dimensions are always kept within model constraints.

        Args:
            image_pil: Input PIL image
            verbose: Whether to print verbose output

        Returns:
            Tuple of (prepared image, original width, original height)
        """
        orig_w, orig_h = image_pil.size
        current_pixels = orig_w * orig_h

        if current_pixels <= 0:
            scale = 1.0
            reason = "model-compatible dimensions"
        elif self.upscale_small_crops:
            target_pixels = 1_048_576  # 2^20 = 1024x1024
            scale = math.sqrt(target_pixels / current_pixels)
            reason = f"~1MP, multiples of {self.RESOLUTION_MULTIPLE}"
        elif current_pixels > self.MAX_INFERENCE_PIXELS:
            scale = math.sqrt(self.MAX_INFERENCE_PIXELS / current_pixels)
            reason = f"4MP cap, multiples of {self.RESOLUTION_MULTIPLE}"
        else:
            scale = 1.0
            reason = "model-compatible dimensions"
        if current_pixels > 0:
            # Fit the longest side here; clamping each side later would distort the shape
            scale = min(
                scale, self.MAX_RESOLUTION / orig_w, self.MAX_RESOLUTION / orig_h
            )

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_w = self._quantize_dimension(new_w)
        new_h = self._quantize_dimension(new_h)
        while new_w * new_h > self.MAX_INFERENCE_PIXELS:
            if new_w >= new_h and new_w > self.MIN_RESOLUTION:
                new_w -= self.RESOLUTION_MULTIPLE
            elif new_h > self.MIN_RESOLUTION:
                new_h -= self.RESOLUTION_MULTIPLE
            else:
                break

        if (new_w, new_h) != (orig_w, orig_h):
            log_message(
                f"  - Scaling {orig_w}x{orig_h} -> {new_w}x{new_h} ({reason})",
                verbose=verbose,
            )
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image_pil, orig_w, orig_h

    def _cache_options(
        self,
        strict_mask_clipping: bool,
        composite_clip_bbox: tuple[int, int, int, int] | None,
        ocr_params: dict | None,
    ) -> dict:
        options = {}
        if strict_mask_clipping:
            options["strict_clip"] = True
        if composite_clip_bbox is not None:
            options["clip_bbox"] = tuple(composite_clip_bbox)
        if ocr_params:
            options.update(ocr_params)
        return options

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        seed: int = 1,
        verbose: bool = False,
        strict_mask_clipping: bool = False,
        composite_clip_bbox: tuple[int, int, int, int] | None = None,
        ocr_params: dict | None = None,
    ) -> Image.Image:
        """Inpaint a specific mask region in the image.

        Args:
            image_pil: PIL Image to inpaint
            mask_np: Numpy mask array (H, W) with True for areas to inpaint
            seed: Random seed for inference
            verbose: Whether to print verbose output
            strict_mask_clipping: When True, ensure compositing is limited to the
                original mask extent
            composite_clip_bbox: Optional (x1, y1, x2, y2) bbox to clip the final
                composite mask to
            ocr_params: Optional OCR parameters dict for cache key generation

        Returns:
            PIL.Image: The inpainted image
        """

        mask_np = np.asarray(mask_np)
        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        if not np.any(mask_np):
            return image_pil

        log_message(f"  - {self.label} inpainting...", verbose=verbose)

        ys, xs = np.where(mask_np)
        if len(ys) == 0 or len(xs) == 0:
            return image_pil

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        padding_pixels = int(max(bbox_width, bbox_height) * CONTEXT_PADDING_RATIO)
        padding = int(
            min(padding_pixels, MAX_CONTEXT_PADDING) * self.PADDING_MULTIPLIER
        )

        blur_radius = int(max(bbox_width, bbox_height) * BLUR_SCALE_FACTOR)
        blur_radius = max(MIN_BLUR_RADIUS, min(blur_radius, MAX_BLUR_RADIUS))

        img_h, img_w = mask_np.shape
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(img_w, x_max + 1 + padding)
        y2 = min(img_h, y_max + 1 + padding)
        x1, y1, x2, y2 = self._expand_bounds_to_min_size(x1, y1, x2, y2, img_w, img_h)

        # Only snap to the patch grid here; _prepare_image_for_inference caps the size
        width = min(self._snap_dimension(x2 - x1), img_w)
        height = min(self._snap_dimension(y2 - y1), img_h)

        if x1 + width > img_w:
            x1 = max(0, img_w - width)
        if y1 + height > img_h:
            y1 = max(0, img_h - height)
        x2 = x1 + width
        y2 = y1 + height
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            log_message(
                f"  - Region has invalid size ({width}x{height}), skipping",
                verbose=verbose,
            )
            return image_pil

        log_message(
            f"  - Processing region at ({x1}, {y1}) size {width}x{height}",
            verbose=verbose,
        )

        image_cropped_pil = image_pil.crop((x1, y1, x2, y2))
        mask_crop_np = mask_np[y1:y2, x1:x2]

        cache_params = {
            "bbox": (x1, y1, width, height),
            "padding": padding,
            "blur": blur_radius,
            "model": self.model_key,
            "lum_corr": self.luminance_correction,
            "upscale_small": self.upscale_small_crops,
            "max_pixels": self.MAX_INFERENCE_PIXELS,
            "min_size": (self.MIN_RESOLUTION, self.MIN_RESOLUTION),
            "backend": self.backend,
            **self._cache_options(
                strict_mask_clipping, composite_clip_bbox, ocr_params
            ),
        }
        cache_key, cached_patch = _lookup_inpaint_patch(
            self.cache,
            image_cropped_pil,
            mask_crop_np,
            seed,
            self.num_inference_steps,
            0.0,  # residual_diff_threshold is Kontext/Nunchaku only
            self.GUIDANCE_SCALE,
            self.PROMPT,
            cache_params,
        )

        mask_for_composite = _feathered_crop_mask(mask_crop_np, blur_radius)

        if strict_mask_clipping:
            original_mask_crop = torch.from_numpy(mask_crop_np.astype(np.float32))
            mask_for_composite = mask_for_composite * original_mask_crop

        if composite_clip_bbox is not None:
            mask_for_composite = _clip_composite_mask(
                mask_for_composite,
                composite_clip_bbox,
                (img_w, img_h),
                (x1, y1, width, height),
            )

        patch_pil = cached_patch

        if patch_pil is None:
            inference_image, _, _ = self._prepare_image_for_inference(
                image_cropped_pil,
                verbose=verbose,
            )
            inference_w, inference_h = inference_image.size

            if inference_image.mode == "RGBA":
                inference_image = inference_image.convert("RGB")

            log_message("  - Running inference...", verbose=verbose)

            with self.manager.flux_inference_lock:
                raise_if_cancelled()  # Jobs queued on the lock skip their inference
                if self.PARKS_TEXT_SAM:
                    self.manager.park_text_sam(verbose=verbose)
                self.load_models()

                if self.pipeline is None:
                    log_message(
                        f"Warning: {self.label} pipeline unavailable.",
                        always_print=True,
                    )
                    return image_pil

                generated_patch_pil = self._generate_patch(
                    inference_image,
                    mask_crop_np,
                    inference_w,
                    inference_h,
                    seed,
                    verbose,
                )

            raise_if_cancelled()  # An interrupted run returns a half-denoised patch
            if (inference_w, inference_h) != (width, height):
                patch_pil = generated_patch_pil.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            else:
                patch_pil = generated_patch_pil

            if self.luminance_correction:
                patch_pil = _match_luminance(
                    generated_pil=patch_pil,
                    original_crop_pil=image_cropped_pil,
                    mask_crop_np=mask_crop_np,
                    verbose=verbose,
                )

        composited_pil = _paste_patch(image_pil, patch_pil, mask_for_composite, x1, y1)

        if (
            self.cache.should_use_inpaint_cache(seed)
            and cache_key is not None
            and cached_patch is None
        ):
            self.cache.set_inpainted_image(cache_key, patch_pil)

        return composited_pil


class LamaInpainter(_CropInpainter):
    """Inpainter using the manga-finetuned Big-LaMa, a fast non-diffusion model."""

    model_key = "lama_manga"
    label = "LaMa Manga"
    # Small enough to share the device with the text segmentation model
    PARKS_TEXT_SAM = False

    def __init__(
        self,
        device: torch.device | None = None,
        luminance_correction: bool = True,
        verbose: bool = False,
    ):
        # LaMa fills from the crop's own context, so it runs at native resolution
        super().__init__(
            backend="torch",
            device=device,
            num_inference_steps=1,
            luminance_correction=luminance_correction,
            upscale_small_crops=False,
            verbose=verbose,
        )

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        *args,
        composite_clip_bbox: tuple[int, int, int, int] | None = None,
        **kwargs,
    ) -> Image.Image:
        # LaMa invents every masked pixel, so mask margin past the text box would
        # erase art that the diffusion models redraw and the composite discards
        if composite_clip_bbox is not None:
            x1, y1, x2, y2 = composite_clip_bbox
            x1, y1 = max(0, x1), max(0, y1)
            clipped = np.zeros(np.shape(mask_np), dtype=bool)
            clipped[y1:y2, x1:x2] = np.asarray(mask_np, dtype=bool)[y1:y2, x1:x2]
            mask_np = clipped
        return super().inpaint_mask(
            image_pil,
            mask_np,
            *args,
            composite_clip_bbox=composite_clip_bbox,
            **kwargs,
        )

    def load_models(self):
        if self.pipeline is None:
            self.pipeline = self.manager.load_lama_manga(verbose=self.verbose)

    def unload_models(self):
        self.pipeline = None
        self.manager.unload_lama_manga()

    def _generate_patch(
        self,
        image: Image.Image,
        mask: np.ndarray,
        width: int,
        height: int,
        seed: int,
        verbose: bool,
    ) -> Image.Image:
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        device = self.pipeline.device
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)[None].to(device)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
        with torch.inference_mode():
            output = self.pipeline(image_t, mask_t)
        output_np = output[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return Image.fromarray((output_np * 255).round().astype(np.uint8))


