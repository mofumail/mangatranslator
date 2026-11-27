"""Local OCR models reading crops: manga-ocr and PaddleOCR-VL 1.6."""

from PIL import Image

from mangatl.common.log import log_message
from mangatl.engine import Model


def extract_text_with_manga_ocr(images: list[Image.Image], models, verbose: bool = False) -> list[str]:
    """Extract text from images using manga-ocr library.

    Args:
        images: List of PIL Images to process
        verbose: Whether to print verbose output

    Returns:
        List of extracted text strings (one per image). Returns [OCR FAILED] on errors.
    """
    if not images:
        return []

    try:
        with models.use(Model.MANGA_OCR):  # load errors fail the whole list
            pass

        extracted_texts = []
        for i, img in enumerate(images):
            try:
                if img is None:
                    log_message(
                        f"Image {i + 1} is None (decode failure), skipping",
                        always_print=True,
                    )
                    extracted_texts.append("[OCR FAILED]")
                    continue

                log_message(
                    f"Processing image {i + 1}/{len(images)} with manga-ocr",
                    verbose=verbose,
                )
                with models.use(Model.MANGA_OCR) as manga_ocr:
                    text = manga_ocr(img)

                extracted_texts.append(text.strip() if text else "")

            except Exception as e:
                log_message(f"manga-ocr failed for image {i + 1}: {e}", always_print=True)
                extracted_texts.append("[OCR FAILED]")

        return extracted_texts

    except Exception as e:
        log_message(f"Error with manga-ocr: {e}", always_print=True)
        return ["[OCR FAILED]"] * len(images)


def _get_image_size_value(size_config, key: str):
    if isinstance(size_config, dict):
        return size_config.get(key)
    return getattr(size_config, key, None)


def _get_paddle_ocr_vl_size(processor, max_pixels: int) -> dict:
    image_processor = processor.image_processor
    size_config = getattr(image_processor, "size", {}) or {}

    shortest_edge = getattr(image_processor, "min_pixels", None)
    if shortest_edge is None:
        shortest_edge = _get_image_size_value(size_config, "shortest_edge")
    if shortest_edge is None:
        shortest_edge = _get_image_size_value(size_config, "min_pixels")
    if shortest_edge is None:
        shortest_edge = 28 * 28 * 130

    return {
        "shortest_edge": shortest_edge,
        "longest_edge": max_pixels,
    }


def extract_text_with_paddle_ocr_vl(images: list[Image.Image], models, verbose: bool = False) -> list[str]:
    """Extract text from images using PaddleOCR-VL-1.6.

    Args:
        images: List of PIL Images to process (RGB)
        verbose: Whether to print verbose output

    Returns:
        List of extracted text strings (one per image). Returns [OCR FAILED] on errors.
    """
    if not images:
        return []

    try:
        with models.use(Model.PADDLE_OCR_VL) as (processor, _):  # load errors fail the whole list
            image_size = _get_paddle_ocr_vl_size(processor, 1280 * 28 * 28)

        extracted_texts = []
        for i, img in enumerate(images):
            try:
                if img is None:
                    log_message(
                        f"Image {i + 1} is None (decode failure), skipping",
                        always_print=True,
                    )
                    extracted_texts.append("[OCR FAILED]")
                    continue

                log_message(
                    f"Processing image {i + 1}/{len(images)} with PaddleOCR-VL-1.6",
                    verbose=verbose,
                )

                with models.use(Model.PADDLE_OCR_VL) as (processor, model):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": "OCR:"},
                            ],
                        }
                    ]
                    inputs = processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        processor_kwargs={
                            "images_kwargs": {
                                "size": image_size,
                            },
                        },
                    ).to(model.device)

                    # The checkpoint's generation config ships use_cache=False
                    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)

                    text = processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])
                extracted_texts.append(text.strip() if text else "")

            except Exception as e:
                log_message(
                    f"PaddleOCR-VL-1.6 failed for image {i + 1}: {e}",
                    always_print=True,
                )
                extracted_texts.append("[OCR FAILED]")

        return extracted_texts

    except Exception as e:
        log_message(f"Error with PaddleOCR-VL-1.6: {e}", always_print=True)
        return ["[OCR FAILED]"] * len(images)
