import contextlib
import copy
import os
import traceback
import warnings
from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, Siglip2VisionModel, Siglip2VisionConfig
from transformers.activations import ACT2FN

try:
    from .configuration_hayai import HayaiConfig
except ImportError:
    from configuration_hayai import HayaiConfig


# =============================================================================
# Capability probes (done once at import)
# =============================================================================
def _probe_enable_gqa() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 8)
        k = torch.zeros(1, 1, 1, 8)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except Exception:
        return False


_HAS_GQA = _probe_enable_gqa()
_HAS_RMS = hasattr(F, "rms_norm")


def _rms(x: torch.Tensor, w, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm; w may be None. Result dtype == x dtype (callers pass fp32 residuals)."""
    if w is not None and w.dtype != x.dtype:
        x = x.float()
    if _HAS_RMS:
        return F.rms_norm(x, (x.shape[-1],), w, eps)
    xf = x.float()
    out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return out if w is None else out * w


def _sdpa_gqa(q, k, v, mask=None):
    """q: (b,hq,s,d)  k,v: (b,hkv,s,d). mask: bool, True = attend."""
    hq, hkv = q.shape[1], k.shape[1]
    if hq == hkv:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if _HAS_GQA and q.device.type != "cpu":   # CPU: repeat -> plain flash kernel is the safe fast path
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    rep = hq // hkv
    return F.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)


def _P(t):
    """Dense, detached view of a parameter. Tensor subclasses (e.g. torchao int8 weights) are dequantized."""
    if t is None:
        return None
    if type(t) not in (torch.Tensor, nn.Parameter):
        dq = getattr(t, "dequantize", None)
        if callable(dq):
            t = dq()
    return t.detach()


def _is_quantized(model) -> bool:
    return any(type(p) not in (torch.Tensor, nn.Parameter) for p in model.decoder.parameters())


# =============================================================================
# RoPE (fully vectorised: no per-sample python loop, no .item(), no host syncs)
# =============================================================================
@lru_cache(maxsize=None)
def _inv_freq(d_axis: int, theta: float, device_str: str, inference: bool = False) -> torch.Tensor:
    dev = torch.device(device_str)
    return 1.0 / (theta ** (torch.arange(0, d_axis, 2, device=dev).float() / d_axis))


def compute_batch_2d_mrope_freqs(spatial_shapes: torch.Tensor, m_vision: int, n_text: int,
                                 d_head: int = 64, theta: float = 10000.0, device="cuda"):
    """spatial_shapes: (b,2) = post-projector (h, w). Returns cos, sin of shape (b, m+n, d_head//2).
    Padded vision slots get angle 0 (cos=1, sin=0), text positions start at 0 -- same as before."""
    dev = torch.device(device)
    b = spatial_shapes.shape[0]
    d_axis = d_head // 2
    freqs = _inv_freq(d_axis, theta, str(dev), torch.is_inference_mode_enabled())

    ss = spatial_shapes.to(dev)
    h, w = ss[:, 0:1], ss[:, 1:2]
    p = torch.arange(m_vision, device=dev)[None]
    wc = w.clamp(min=1)
    y = (p // wc).float()
    x = (p % wc).float()
    ang = torch.cat([y[..., None] * freqs, x[..., None] * freqs], dim=-1)
    ang = torch.where((p < h * w)[..., None], ang, torch.zeros_like(ang))

    t = torch.arange(n_text, device=dev, dtype=torch.float32)
    t_ang = torch.outer(t, freqs)
    t_ang = torch.cat([t_ang, t_ang], dim=-1)

    ang = torch.cat([ang, t_ang[None].expand(b, -1, -1)], dim=1)
    return ang.cos(), ang.sin()


def apply_rotary_emb_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (b,s,h,d). cos/sin: (b|1, s|1, d/2). Interleaved-pair rotation in fp32, no complex tensors."""
    dtype = x.dtype
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos.unsqueeze(2), sin.unsqueeze(2)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)
    return out.to(dtype)


def _norm_rope(x, w, cos, sin, out_dtype, eps=1e-6):
    """Fused-by-shape q/k head RMSNorm (+ per-head weight w: (h,d) fp32) followed by RoPE."""
    xf = _rms(x.float(), None, eps) * w
    x1, x2 = xf.unflatten(-1, (-1, 2)).unbind(-1)
    c, s = cos.unsqueeze(2), sin.unsqueeze(2)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2).to(out_dtype)


# =============================================================================
# Modules (parameter names unchanged -> checkpoints still load)
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_RMS:
            return F.rms_norm(x, (self.weight.numel(),), self.weight, self.eps)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class DSCProjector(nn.Module):
    def __init__(self, d_vision: int = 768, d_model: int = 512, downscale_factor: int = 2):
        super().__init__()
        self.downscale = downscale_factor
        unshuffle_dim = d_vision * (downscale_factor ** 2)
        self.norm = nn.LayerNorm(unshuffle_dim)
        self.mlp = nn.Sequential(
            nn.Linear(unshuffle_dim, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.out_norm = RMSNorm(d_model)

    @staticmethod
    def gather_patches(flat: torch.Tensor, base: torch.Tensor, spatial_shapes: torch.Tensor, r: int, bucket: int = 1):
        """Vectorised replacement for the per-sample view/pad/pixel_unshuffle loop.

        flat: (R, d) token table; base: (b,) row offset of each sample (padded: i*m, packed: cumulative).
        Each sample's valid tokens are rows base[i] .. base[i]+h*w (row-major).
        Returns:
          gathered   (b, L, r*r, d)  -- the r*r neighbours of every merged token (replicate-padded
                                         at the bottom/right via index clamping)
          new_shapes (b, 2) on device, lens (b,) on device, lens_cpu (b,), L (int, rounded up to `bucket`)
        Only ONE host transfer (the tiny spatial_shapes tensor) -- skipped if it's already on CPU.
        """
        d = flat.shape[-1]
        b = spatial_shapes.shape[0]
        dev = flat.device
        ss = spatial_shapes.detach().to("cpu", torch.long)
        h_c, w_c = ss[:, 0], ss[:, 1]
        ho_c, wo_c = (h_c + r - 1) // r, (w_c + r - 1) // r
        lens_c = ho_c * wo_c
        L = int(lens_c.max())
        L = max(1, -(-L // bucket) * bucket)

        h, w, ho, wo, lens = (t.to(dev, non_blocking=True) for t in (h_c, w_c, ho_c, wo_c, lens_c))
        p = torch.arange(L, device=dev)[None]
        woc = wo.clamp(min=1)[:, None]
        oy, ox = p // woc, p % woc                                   # (b, L)
        off = torch.arange(r, device=dev)
        sy = torch.minimum(oy[..., None, None] * r + off[:, None], (h - 1)[:, None, None, None])
        sx = torch.minimum(ox[..., None, None] * r + off[None, :], (w - 1)[:, None, None, None])
        idx = (sy.clamp(min=0) * w[:, None, None, None] + sx.clamp(min=0))   # (b, L, r, r)
        idx = idx.reshape(b, L * r * r) + base[:, None]
        g = flat.index_select(0, idx.reshape(-1)).view(b, L, r * r, d)
        new_shapes = torch.stack([ho, wo], dim=1)
        return g, new_shapes, lens, lens_c, L

    def forward(self, visual_features: torch.Tensor, spatial_shapes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bb, mm, _ = visual_features.shape
        base = torch.arange(bb, device=visual_features.device) * mm
        g, new_shapes, _, _, L = self.gather_patches(
            visual_features.reshape(bb * mm, -1), base, spatial_shapes, self.downscale)
        b = g.shape[0]
        # (b,L,rr,d) -> (b,L,d,rr): channel index c*rr + k, identical to F.pixel_unshuffle ordering
        tokens = g.transpose(2, 3).reshape(b, L, -1)
        out = self.out_norm(self.mlp(self.norm(tokens)))
        return out, new_shapes


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GroupedQueryAttention(nn.Module):
    """Training / reference path (no KV cache). Inference uses _FastEngine below."""

    def __init__(self, d_model: int, h_q: int = 8, h_kv: int = 2, d_head: int = 64):
        super().__init__()
        self.h_q, self.h_kv, self.d_head = h_q, h_kv, d_head
        self.w_q = nn.Linear(d_model, h_q * d_head, bias=False)
        self.w_k = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_v = nn.Linear(d_model, h_kv * d_head, bias=False)
        self.w_o = nn.Linear(h_q * d_head, d_model, bias=False)
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

    def forward(self, x, mask=None, cos_sin=None, kv_cache=None, layer_idx=None, cache_seqlens=0):
        b, s, _ = x.shape
        q = self.q_norm(self.w_q(x).view(b, s, self.h_q, self.d_head))
        k = self.k_norm(self.w_k(x).view(b, s, self.h_kv, self.d_head))
        v = self.w_v(x).view(b, s, self.h_kv, self.d_head)
        if cos_sin is not None:
            q = apply_rotary_emb_2d(q, *cos_sin)
            k = apply_rotary_emb_2d(k, *cos_sin)
        k_t, v_t = k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            kc, vc = kv_cache[layer_idx]
            kc[:, :, cache_seqlens:cache_seqlens + s] = k_t
            vc[:, :, cache_seqlens:cache_seqlens + s] = v_t
            k_t, v_t = kc[:, :, :cache_seqlens + s], vc[:, :, :cache_seqlens + s]
        ctx = _sdpa_gqa(q.transpose(1, 2), k_t, v_t, mask)
        return self.w_o(ctx.transpose(1, 2).reshape(b, s, -1))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h_q: int, h_kv: int, d_ffn: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, h_q, h_kv)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ffn)
        self.attn_res_scale = nn.Parameter(torch.ones(d_model))
        self.ffn_res_scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x, mask=None, cos_sin=None, kv_cache=None, layer_idx=None, cache_seqlens=0):
        x = x + self.attn_res_scale * self.attn(self.attn_norm(x), mask=mask, cos_sin=cos_sin, kv_cache=kv_cache,
                                                layer_idx=layer_idx, cache_seqlens=cache_seqlens)
        x = x + self.ffn_res_scale * self.ffn(self.ffn_norm(x))
        return x


@lru_cache(maxsize=32)
def _base_block_mask(m_vision: int, n_text: int, device_str: str, inference: bool = False) -> torch.Tensor:
    """Bool (T,T), True = may attend. vision<->vision, text->vision, text->text causal."""
    dev = torch.device(device_str)
    total = m_vision + n_text
    allowed = torch.zeros((total, total), dtype=torch.bool, device=dev)
    allowed[:m_vision, :m_vision] = True
    allowed[m_vision:, :m_vision] = True
    allowed[m_vision:, m_vision:] = torch.tril(torch.ones((n_text, n_text), dtype=torch.bool, device=dev))
    return allowed


class VisualCausalOCRDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, d_vision: int = 768,
                 d_ffn: int = 2048, n_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.projector = DSCProjector(d_vision, d_model, downscale_factor=2)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, 8, 2, d_ffn) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    @staticmethod
    def generate_block_causal_mask(m_vision: int, n_text: int, device, vis_lens: torch.Tensor = None):
        """Bool mask (1|b, 1, T, T); padded vision key slots are masked out when vis_lens is given."""
        base = _base_block_mask(m_vision, n_text, str(device), torch.is_inference_mode_enabled())[None, None]
        if vis_lens is None:
            return base
        pos = torch.arange(m_vision, device=device)[None]
        key_valid = torch.cat([pos < vis_lens[:, None],
                               torch.ones((vis_lens.size(0), n_text), dtype=torch.bool, device=device)], dim=1)
        return base & key_valid[:, None, None, :]

    def forward(self, visual_features, spatial_shapes, text_token_ids):
        vis, new_shapes = self.projector(visual_features, spatial_shapes)
        m_vision = vis.size(1)
        n_text = text_token_ids.size(1)
        vis_lens = new_shapes[:, 0] * new_shapes[:, 1]

        x = torch.cat([vis, self.token_embeddings(text_token_ids).to(vis.dtype)], dim=1)
        mask = self.generate_block_causal_mask(m_vision, n_text, x.device, vis_lens)
        cos_sin = compute_batch_2d_mrope_freqs(new_shapes, m_vision, n_text, d_head=64, device=x.device)
        for layer in self.layers:
            x = layer(x, mask=mask, cos_sin=cos_sin)
        return self.output_head(self.final_norm(x[:, m_vision:]))


# =============================================================================
# Inference engine
# =============================================================================
class _FastVision:
    """Inference-only re-implementation of Siglip2VisionModel (NaFlex), built from its weights.

      * fused QKV GEMM, fused residual adds (in-place), no HF module/mask-expansion overhead
      * bilinear/antialiased position embeddings cached per (h, w) -- the HF module re-runs
        F.interpolate for every sample of every call
      * ragged batches are PACKED (concatenate only the valid patches, per-sample attention with no
        mask -> flash kernel): zero GEMM FLOPs wasted on padding. Near-uniform batches stay padded.
      * returns (flat (R, D) tokens, base (b,) row offsets) so the projector gathers straight from it
    """
    PACK_THRESHOLD = 0.05      # pack when > 5 % of the padded tokens are padding
    POS_CACHE = 128

    def __init__(self, vm_model, dtype):
        vm, cfg = vm_model.vision_model, vm_model.config
        emb = vm.embeddings
        if not isinstance(emb.patch_embedding, nn.Linear):
            raise TypeError("unsupported Siglip2 layout")
        dt = self.dtype = dtype
        self.D, self.H = cfg.hidden_size, cfg.num_attention_heads
        self.hd = self.D // self.H
        self.eps = cfg.layer_norm_eps
        self.act = ACT2FN[cfg.hidden_act]
        c = lambda t: t.detach().to(dt).contiguous()
        self.patch_w, self.patch_b = c(_P(emb.patch_embedding.weight)), c(_P(emb.patch_embedding.bias))
        self.pos = _P(emb.position_embedding.weight).detach().float()
        self.pos_side = int(round(self.pos.shape[0] ** 0.5))
        self.layers = []
        for ly in vm.encoder.layers:
            at = ly.self_attn
            self.layers.append(SimpleNamespace(
                ln1_w=c(_P(ly.layer_norm1.weight)), ln1_b=c(_P(ly.layer_norm1.bias)),
                w_qkv=c(torch.cat([_P(at.q_proj.weight), _P(at.k_proj.weight), _P(at.v_proj.weight)], 0)),
                b_qkv=c(torch.cat([_P(at.q_proj.bias), _P(at.k_proj.bias), _P(at.v_proj.bias)], 0)),
                w_o=c(_P(at.out_proj.weight)), b_o=c(_P(at.out_proj.bias)),
                ln2_w=c(_P(ly.layer_norm2.weight)), ln2_b=c(_P(ly.layer_norm2.bias)),
                fc1_w=c(_P(ly.mlp.fc1.weight)), fc1_b=c(_P(ly.mlp.fc1.bias)),
                fc2_w=c(_P(ly.mlp.fc2.weight)), fc2_b=c(_P(ly.mlp.fc2.bias))))
        self.post_w, self.post_b = c(_P(vm.post_layernorm.weight)), c(_P(vm.post_layernorm.bias))
        self._pos_cache = OrderedDict()

    def pos_embed(self, h: int, w: int) -> torch.Tensor:
        key = (h, w)
        t = self._pos_cache.get(key)
        if t is not None:
            self._pos_cache.move_to_end(key)
            return t
        pe = self.pos.view(self.pos_side, self.pos_side, -1).permute(2, 0, 1)[None]
        t = F.interpolate(pe, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        t = t.reshape(self.D, h * w).t().contiguous().to(self.dtype)
        self._pos_cache[key] = t
        while len(self._pos_cache) > self.POS_CACHE:
            self._pos_cache.popitem(last=False)
        return t

    def forward(self, pixel_values: torch.Tensor, ss_cpu: torch.Tensor):
        dev, dt, D, H, hd, eps = pixel_values.device, self.dtype, self.D, self.H, self.hd, self.eps
        b, n, _ = pixel_values.shape
        shapes = ss_cpu.tolist()
        hw = [h * w for h, w in shapes]
        pad = b * n - sum(hw)
        pack = pad > self.PACK_THRESHOLD * b * n

        mask = None
        if pack:
            x = F.linear(torch.cat([pixel_values[i, :l] for i, l in enumerate(hw)], 0).to(dt),
                         self.patch_w, self.patch_b)                               # (T, D)
            offs = [0]
            for l in hw:
                offs.append(offs[-1] + l)
            for i, (h, w) in enumerate(shapes):
                if hw[i]:
                    x[offs[i]:offs[i] + hw[i]].add_(self.pos_embed(h, w))
            base = torch.tensor(offs[:-1], dtype=torch.long, device=dev)
        else:
            x = F.linear(pixel_values.to(dt), self.patch_w, self.patch_b)          # (b, n, D)
            for i, (h, w) in enumerate(shapes):
                if hw[i]:
                    x[i, :hw[i]].add_(self.pos_embed(h, w))
            base = torch.arange(b, device=dev) * n
            if pad > 0:
                lens = torch.tensor(hw, dtype=torch.long, device=dev)
                mask = (torch.arange(n, device=dev)[None] < lens[:, None])[:, None, None, :]

        for L in self.layers:
            h = F.layer_norm(x, (D,), L.ln1_w, L.ln1_b, eps)
            qkv = F.linear(h, L.w_qkv, L.b_qkv)
            if pack:
                a = torch.empty_like(h)
                off = 0
                for l in hw:
                    if l:
                        q, k, v = qkv[off:off + l].view(l, 3, H, hd).permute(1, 2, 0, 3).unbind(0)
                        a[off:off + l] = F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape(l, D)
                    off += l
            else:
                q, k, v = qkv.view(b, n, 3, H, hd).permute(2, 0, 3, 1, 4).unbind(0)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, n, D)
            x.add_(F.linear(a, L.w_o, L.b_o))
            h = F.layer_norm(x, (D,), L.ln2_w, L.ln2_b, eps)
            x.add_(F.linear(self.act(F.linear(h, L.fc1_w, L.fc1_b)), L.fc2_w, L.fc2_b))

        x = F.layer_norm(x, (D,), self.post_w, self.post_b, eps)
        return x.reshape(-1, D), base


class _HFVision:
    """Fallback if the installed transformers has a different Siglip2 layout."""

    def __init__(self, vm_model, dtype):
        self.dtype = dtype
        self.m = vm_model if next(vm_model.parameters()).dtype == dtype else copy.deepcopy(vm_model).to(dtype)
        self.m.eval()

    def forward(self, pixel_values, ss_cpu):
        dev = pixel_values.device
        b, n, _ = pixel_values.shape
        lens = (ss_cpu[:, 0] * ss_cpu[:, 1]).to(dev)
        am = (torch.arange(n, device=dev)[None] < lens[:, None]).long()
        out = self.m(pixel_values=pixel_values.to(self.dtype), pixel_attention_mask=am,
                     spatial_shapes=ss_cpu.to(dev)).last_hidden_state
        return out.reshape(b * n, -1), torch.arange(b, device=dev) * n


class _FastEngine:
    """Built lazily from a HayaiModel's weights. Holds inference-only fused copies:

      * weights pre-cast to the compute dtype ONCE (no per-call autocast casts)
      * fused QKV (512 -> 768) and fused gate/up (d -> 2*ffn) GEMMs
      * q_norm/k_norm merged into one (hq+hkv, d_head) weight -> one norm + one RoPE per layer
      * projector's first Linear / LayerNorm columns permuted so the gathered patches can be fed
        directly (no transpose copy before the 4d-wide projector)
      * half-precision copy of the vision encoder
    Decode is shape-static (full-length cache + mask) so the whole step can live in a CUDA graph.
    The residual stream stays fp32 (as under the old autocast), matmuls run in `dtype`.
    """

    MAX_STATES = 4

    def __init__(self, model, dtype: torch.dtype):
        dec = model.decoder
        dev = next(model.parameters()).device
        self.device, self.dtype = dev, dtype
        dt = dtype

        try:
            self.vision = _FastVision(model.vision_encoder, dt)
        except Exception:
            self.vision = _HFVision(model.vision_encoder, dt)

        # projector
        pj = dec.projector
        self.r = pj.downscale
        rr = self.r ** 2
        d_v = pj.norm.normalized_shape[0] // rr

        def perm_vec(t):                       # (d*rr) c*rr+k  ->  k*d+c
            return t.detach().view(d_v, rr).t().reshape(-1)

        self.pj_ln_w = perm_vec(_P(pj.norm.weight)).to(dt).contiguous()
        self.pj_ln_b = perm_vec(_P(pj.norm.bias)).to(dt).contiguous()
        self.pj_ln_eps = pj.norm.eps
        w1 = _P(pj.mlp[0].weight).detach()
        self.pj_w1 = w1.view(w1.shape[0], d_v, rr).permute(0, 2, 1).reshape(w1.shape[0], -1).to(dt).contiguous()
        self.pj_w2 = _P(pj.mlp[2].weight).detach().to(dt).contiguous()
        self.pj_out_w = _P(pj.out_norm.weight).detach().float().contiguous()
        self.pj_out_eps = pj.out_norm.eps

        # decoder
        a0 = dec.layers[0].attn
        self.hq, self.hkv, self.dh = a0.h_q, a0.h_kv, a0.d_head
        self.g = self.hq // self.hkv
        self.nqk = self.hq + self.hkv
        self.d_axis = self.dh // 2
        self.emb_w = _P(dec.token_embeddings.weight).detach().float().contiguous()   # fp32 residual stream
        self.layers = []
        for ly in dec.layers:
            at, ff = ly.attn, ly.ffn
            qk_w = torch.cat([_P(at.q_norm.weight).detach().expand(self.hq, -1),
                              _P(at.k_norm.weight).detach().expand(self.hkv, -1)], 0).float().contiguous()
            self.layers.append(SimpleNamespace(
                attn_norm_w=_P(ly.attn_norm.weight).detach().float().contiguous(),
                w_qkv=torch.cat([_P(at.w_q.weight), _P(at.w_k.weight), _P(at.w_v.weight)], 0).detach().to(dt).contiguous(),
                qk_norm_w=qk_w,
                w_o=_P(at.w_o.weight).detach().to(dt).contiguous(),
                attn_res=ly.attn_res_scale.detach().float().contiguous(),
                ffn_norm_w=_P(ly.ffn_norm.weight).detach().float().contiguous(),
                w_gu=torch.cat([_P(ff.w_gate.weight), _P(ff.w_up.weight)], 0).detach().to(dt).contiguous(),
                w_down=_P(ff.w_down.weight).detach().to(dt).contiguous(),
                ffn_res=ly.ffn_res_scale.detach().float().contiguous(),
            ))
        self.final_norm_w = _P(dec.final_norm.weight).detach().float().contiguous()
        self.head_w = _P(dec.output_head.weight).detach().to(dt).contiguous()

        self._states = OrderedDict()
        self._cap_stream = None          # one dedicated stream: warm-up AND capture happen on it
        self._graph_failed = False

    # ------------------------------------------------------------------ pieces
    def _ffn(self, L, x):
        xn = _rms(x, L.ffn_norm_w).to(self.dtype)
        g, u = F.linear(xn, L.w_gu).chunk(2, dim=-1)
        return x + L.ffn_res * F.linear(F.silu(g) * u, L.w_down)

    def _project(self, gathered):
        b, L = gathered.shape[:2]
        x = gathered.reshape(b, L, -1)
        x = F.layer_norm(x, (x.shape[-1],), self.pj_ln_w, self.pj_ln_b, self.pj_ln_eps)
        x = F.linear(F.gelu(F.linear(x, self.pj_w1)), self.pj_w2)
        return _rms(x.float(), self.pj_out_w, self.pj_out_eps)             # fp32 residual stream

    @staticmethod
    def _text_tables(n: int, d_axis: int, dev):
        freqs = _inv_freq(d_axis, 10000.0, str(dev), torch.is_inference_mode_enabled())
        a = torch.outer(torch.arange(n, device=dev, dtype=torch.float32), freqs)
        a = torch.cat([a, a], dim=-1)
        return a.cos().contiguous(), a.sin().contiguous()

    # ------------------------------------------------------------------ state / graph
    def _get_state(self, b, L_total, max_new, eos, pad, use_graph):
        key = (b, L_total, max_new, eos, pad, use_graph)
        st = self._states.get(key)
        if st is not None:
            self._states.move_to_end(key)
            return st
        dev, dt = self.device, self.dtype
        kv = [(torch.zeros(b, self.hkv, L_total, self.dh, dtype=dt, device=dev),
               torch.zeros(b, self.hkv, L_total, self.dh, dtype=dt, device=dev)) for _ in self.layers]
        cos_t, sin_t = self._text_tables(max_new + 1, self.d_axis, dev)
        st = SimpleNamespace(
            kv=kv, tok=torch.zeros(b, dtype=torch.long, device=dev),
            step=torch.zeros(1, dtype=torch.long, device=dev),
            cidx=torch.zeros(1, dtype=torch.long, device=dev),
            unfinished=torch.ones(b, dtype=torch.bool, device=dev),
            out=torch.full((b, max_new + 1), pad, dtype=torch.long, device=dev),
            base_valid=torch.ones(b, L_total, dtype=torch.bool, device=dev),
            ar=torch.arange(L_total, device=dev),
            txt_cos=cos_t, txt_sin=sin_t, eos=eos, pad=pad, graph=None, need_mask=True)
        if use_graph:
            try:
                self._capture(st)
            except Exception as e:                       # never let graph capture take generation down
                self._graph_failed = True
                st.graph = None
                torch.cuda.synchronize()                 # re-raises if the context is genuinely broken
                torch.cuda.empty_cache()
                warnings.warn(f"CUDA-graph capture failed ({type(e).__name__}: {str(e)[:200]}); "
                              "falling back to eager decoding.")
        self._states[key] = st
        while len(self._states) > self.MAX_STATES:
            self._states.popitem(last=False)
        return st

    @staticmethod
    def _reset_dummy(st):
        st.step.zero_(); st.cidx.zero_(); st.tok.zero_()
        st.unfinished.fill_(True); st.base_valid.fill_(True); st.out.fill_(st.pad)

    def _capture(self, st):
        # cuBLAS keeps its handle/workspace per (handle, stream). Warming up on one stream and capturing
        # on another makes cuBLAS allocate/initialise *during* capture, which fails for the big GEMMs.
        # So: warm up and capture on the SAME stream, with a private pool and thread-local capture mode.
        if self._cap_stream is None:
            self._cap_stream = torch.cuda.Stream()
        s = self._cap_stream
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            for _ in range(3):
                self._reset_dummy(st)
                self.decode_step(st)
        s.synchronize()
        self._reset_dummy(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"):
            self.decode_step(st)
        st.graph = g

    # ------------------------------------------------------------------ decode step (static shapes)
    def decode_step(self, st, kv_len=None):
        """kv_len=None: static full-length (CUDA-graph) mode. kv_len=int: eager mode, attend only to the live cache."""
        b = st.tok.shape[0]
        hq, hkv, dh, g, nqk = self.hq, self.hkv, self.dh, self.g, self.nqk
        dt = self.dtype

        x = F.embedding(st.tok.unsqueeze(1), self.emb_w)                       # (b,1,d) fp32
        cos = st.txt_cos.index_select(0, st.step).view(1, 1, -1)
        sin = st.txt_sin.index_select(0, st.step).view(1, 1, -1)
        if kv_len is None:
            mask = (st.base_valid & (st.ar[None, :] <= st.cidx))[:, None, None, :]  # (b,1,1,L)
            sl = slice(None)
        else:
            mask = st.base_valid[:, :kv_len][:, None, None, :] if st.need_mask else None
            sl = slice(0, kv_len)

        for li, L in enumerate(self.layers):
            xn = _rms(x, L.attn_norm_w).to(dt)
            qkv = F.linear(xn, L.w_qkv)
            qk = _norm_rope(qkv[..., :nqk * dh].unflatten(-1, (nqk, dh)), L.qk_norm_w, cos, sin, dt)
            v = qkv[..., nqk * dh:].unflatten(-1, (hkv, dh))
            kc, vc = st.kv[li]
            kc.index_copy_(2, st.cidx, qk[:, :, hq:].transpose(1, 2))
            vc.index_copy_(2, st.cidx, v.transpose(1, 2))
            # GQA without repeat/enable_gqa: the g query heads of a kv head become g "query rows"
            qg = qk[:, 0, :hq].reshape(b, hkv, g, dh)
            o = F.scaled_dot_product_attention(qg, kc[:, :, sl], vc[:, :, sl], attn_mask=mask).reshape(b, 1, hq * dh)
            x = x + L.attn_res * F.linear(o, L.w_o)
            x = self._ffn(L, x)

        logits = F.linear(_rms(x, self.final_norm_w).to(dt)[:, 0], self.head_w)  # (b,V)
        nxt = logits.argmax(-1).masked_fill(~st.unfinished, st.pad)
        st.out.index_copy_(1, st.step + 1, nxt.unsqueeze(1))
        st.unfinished.copy_(st.unfinished & (nxt != st.eos) & (nxt != st.pad))
        st.tok.copy_(nxt)
        st.step.add_(1)
        st.cidx.add_(1)

    # ------------------------------------------------------------------ prefill
    def prefill(self, x, vis_valid, need_mask, cos, sin, st, m):
        """x: (b, m+1, d) fp32 = [vision | BOS]. Vision rows never attend to text, so the attention
        splits in two calls with tiny broadcast masks (and NO mask -> flash kernel when no padding).
        The last layer only computes Q / attention / FFN for the BOS row (K,V still for all rows)."""
        b, s, _ = x.shape
        hq, hkv, dh, g, nqk = self.hq, self.hkv, self.dh, self.g, self.nqk
        qd = hq * dh
        dt = self.dtype
        vmask = allmask = None
        if need_mask:
            vmask = vis_valid[:, None, None, :]
            allmask = torch.cat([vis_valid, torch.ones((b, 1), dtype=torch.bool, device=x.device)], 1)[:, None, None, :]
        n = len(self.layers)

        for li, L in enumerate(self.layers):
            last = li == n - 1
            xn = _rms(x, L.attn_norm_w).to(dt)
            if not last:
                qkv = F.linear(xn, L.w_qkv)
                qk = _norm_rope(qkv[..., :nqk * dh].unflatten(-1, (nqk, dh)), L.qk_norm_w, cos, sin, dt)
                v = qkv[..., nqk * dh:].unflatten(-1, (hkv, dh))
                q, k = qk[:, :, :hq], qk[:, :, hq:]
            else:
                kv = F.linear(xn, L.w_qkv[qd:])
                q = _norm_rope(F.linear(xn[:, -1:], L.w_qkv[:qd]).unflatten(-1, (hq, dh)),
                               L.qk_norm_w[:hq], cos[:, -1:], sin[:, -1:], dt)
                k = _norm_rope(kv[..., :hkv * dh].unflatten(-1, (hkv, dh)),
                               L.qk_norm_w[hq:], cos, sin, dt)
                v = kv[..., hkv * dh:].unflatten(-1, (hkv, dh))

            k_t, v_t = k.transpose(1, 2), v.transpose(1, 2)                    # (b,hkv,s,dh)
            kc, vc = st.kv[li]
            kc[:, :, :s].copy_(k_t)
            vc[:, :, :s].copy_(v_t)

            q_b = q[:, -1].reshape(b, hkv, g, dh)
            o_b = F.scaled_dot_product_attention(q_b, k_t, v_t, attn_mask=allmask).reshape(b, 1, qd)
            if not last:
                o_v = _sdpa_gqa(q[:, :m].transpose(1, 2), k_t[:, :, :m], v_t[:, :, :m], vmask)
                o = torch.cat([o_v.transpose(1, 2).reshape(b, m, qd), o_b], dim=1)
            else:
                o = o_b
                x = x[:, -1:]
            x = x + L.attn_res * F.linear(o, L.w_o)
            x = self._ffn(L, x)

        return F.linear(_rms(x, self.final_norm_w).to(dt)[:, 0], self.head_w)   # (b,V)

    # ------------------------------------------------------------------ full generate
    def run(self, pixel_values, pixel_attention_mask, spatial_shapes, bos, eos, pad, max_new, use_graph):
        dev, dt = self.device, self.dtype
        use_graph = use_graph and not self._graph_failed
        ss_cpu = spatial_shapes.detach().to("cpu", torch.long)            # the single unavoidable sync
        n_valid = int((ss_cpu[:, 0] * ss_cpu[:, 1]).max())

        # valid patches are a prefix of each sample (pixel_attention_mask is derived from spatial_shapes)
        flat, base = self.vision.forward(pixel_values[:, :n_valid], ss_cpu)

        bucket = 32 if use_graph else 1                                    # fewer distinct graph keys
        gathered, new_shapes, lens, lens_c, m = DSCProjector.gather_patches(flat, base, ss_cpu, self.r, bucket)
        need_mask = bool((lens_c < m).any())                               # CPU-only check
        b = ss_cpu.shape[0]

        vis_emb = self._project(gathered)
        bos_emb = F.embedding(torch.full((b, 1), bos, dtype=torch.long, device=dev), self.emb_w)
        x = torch.cat([vis_emb, bos_emb], dim=1)
        cos, sin = compute_batch_2d_mrope_freqs(new_shapes, m, 1, d_head=2 * self.d_axis, device=dev)

        st = self._get_state(b, m + max_new + 1, max_new, eos, pad, use_graph)
        vis_valid = torch.arange(m, device=dev)[None] < lens[:, None]
        st.base_valid.fill_(True)
        st.base_valid[:, :m] = vis_valid
        st.out.fill_(pad)
        st.out[:, 0] = bos
        st.step.fill_(1)
        st.cidx.fill_(m + 1)
        st.need_mask = need_mask

        logits = self.prefill(x, vis_valid, need_mask, cos, sin, st, m)
        nxt = logits.argmax(-1)
        st.out[:, 1] = nxt
        st.unfinished.copy_((nxt != eos) & (nxt != pad))
        st.tok.copy_(nxt)

        for i in range(1, max_new):
            if (i - 1) % 16 == 0 and not bool(st.unfinished.any()):         # sync only every 16 steps
                break
            if st.graph is not None:
                st.graph.replay()
            else:
                self.decode_step(st, kv_len=m + i + 1)
        return st.out.cpu()


# =============================================================================
# Model
# =============================================================================
@lru_cache(maxsize=1)
def _cpu_has_fast_bf16() -> bool:
    """Native bf16 matmul (AMX / AVX512-BF16). Without it bf16 on CPU is *slower* than fp32."""
    tcpu = getattr(torch, "cpu", None)
    for name in ("_is_amx_bf16_supported", "_is_amx_tile_supported", "_is_avx512_bf16_supported"):
        fn = getattr(tcpu, name, None)
        if fn is not None:
            try:
                if fn():
                    return True
            except Exception:
                pass
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return any(t in flags for t in ("amx_bf16", "amx_tile", "avx512_bf16"))
    except Exception:
        return False


def _ambient_autocast_dtype(device_type: str):
    """dtype of an enclosing torch.autocast on this device type, or None."""
    if device_type not in ("cpu", "cuda"):
        return None
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        return None
    except Exception:                                      # older torch: per-device functions
        try:
            if device_type == "cpu" and torch.is_autocast_cpu_enabled():
                return torch.get_autocast_cpu_dtype()
            if device_type == "cuda" and torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
        except Exception:
            pass
        return None


def _default_dtype(device: torch.device) -> torch.dtype:
    """Best default per device: an enclosing autocast dtype wins; else fp16 on GPU/MPS, bf16 on CPUs with
    native bf16, fp32 everywhere else."""
    amb = _ambient_autocast_dtype(device.type)
    if amb is not None:
        return amb
    if device.type in ("cuda", "mps"):
        return torch.float16
    if device.type == "cpu" and _cpu_has_fast_bf16():
        return torch.bfloat16
    return torch.float32


def _no_autocast(device_type: str):
    """The engine manages dtypes itself; autocast on top would only re-cast weights on every call."""
    if device_type in ("cpu", "cuda"):
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def _is_oom(e: Exception) -> bool:
    return isinstance(e, MemoryError) or type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


class HayaiModel(PreTrainedModel):
    config_class = HayaiConfig

    def __init__(self, config: HayaiConfig):
        super().__init__(config)
        vcfg = getattr(config, "vision_config", None)       # optional: avoids a hub round-trip on init
        if isinstance(vcfg, dict):
            vcfg = Siglip2VisionConfig(**vcfg)
        if vcfg is None:
            vcfg = Siglip2VisionConfig.from_pretrained("google/siglip2-base-patch16-naflex")
        self.vision_encoder = Siglip2VisionModel(vcfg)
        self.decoder = VisualCausalOCRDecoder(
            vocab_size=config.vocab_size, d_model=config.d_model, d_vision=config.d_vision,
            d_ffn=config.d_ffn, n_layers=config.n_layers)
        self._engines = {}
        self.post_init()
        self.tie_weights()

    # engines hold fused/cast copies of the weights -> drop them whenever weights/device may change
    def reset_inference_cache(self):
        self.__dict__["_engines"] = {}
        self.__dict__["_failed_dtypes"] = set()

    def _apply(self, fn, *args, **kwargs):
        self.reset_inference_cache()
        return super()._apply(fn, *args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.reset_inference_cache()
        return super().load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        if mode:
            self.reset_inference_cache()
        return super().train(mode)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, text_token_ids):
        vis = self.vision_encoder(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                                  spatial_shapes=spatial_shapes)
        return self.decoder(vis.last_hidden_state, spatial_shapes, text_token_ids)

    @torch.no_grad()
    def _reference_generate(self, pixel_values, pixel_attention_mask, spatial_shapes,
                            bos, eos, pad, max_new, device):
        """Module-based KV-cache decoding (the original algorithm + padding-mask fix). Slower than the fast
        engine, but it runs through the model's own modules, so it honours torchao quantization,
        torch.compile, etc. that were applied to them."""
        dt_ = device.type
        amp = _default_dtype(device)
        ctx = (torch.autocast(device_type=dt_, dtype=amp, enabled=(amp != torch.float32))
               if dt_ in ("cpu", "cuda") else contextlib.nullcontext())
        dec = self.decoder
        b = pixel_values.size(0)
        with ctx:
            vis = self.vision_encoder(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                                      spatial_shapes=spatial_shapes).last_hidden_state
            vis_emb, new_shapes = dec.projector(vis, spatial_shapes)
            m = vis_emb.size(1)
            vis_lens = new_shapes[:, 0] * new_shapes[:, 1]
            bos_t = torch.full((b, 1), bos, dtype=torch.long, device=device)
            x = torch.cat([vis_emb, dec.token_embeddings(bos_t)], dim=1)
            mask = dec.generate_block_causal_mask(m, 1, device, vis_lens)
            cos, sin = compute_batch_2d_mrope_freqs(new_shapes, m, 1, d_head=64, device=device)

            a0 = dec.layers[0].attn
            total = m + max_new + 1
            cache = {i: (torch.zeros(b, a0.h_kv, total, a0.d_head, dtype=x.dtype, device=device),
                         torch.zeros(b, a0.h_kv, total, a0.d_head, dtype=x.dtype, device=device))
                     for i in range(len(dec.layers))}
            for i, layer in enumerate(dec.layers):
                x = layer(x, mask=mask, cos_sin=(cos, sin), kv_cache=cache, layer_idx=i, cache_seqlens=0)
            seqlen = x.size(1)
            nxt = dec.output_head(dec.final_norm(x[:, -1:]))[:, -1].argmax(-1)

            out = torch.full((b, max_new + 1), pad, dtype=torch.long, device=device)
            out[:, 0] = bos
            out[:, 1] = nxt
            unfinished = (nxt != eos) & (nxt != pad)

            freqs = _inv_freq(32, 10000.0, str(device), torch.is_inference_mode_enabled())
            ang = torch.outer(torch.arange(max_new + 1, device=device, dtype=torch.float32), freqs)
            ang = torch.cat([ang, ang], dim=-1)
            cos_t, sin_t = ang.cos(), ang.sin()
            key_valid = torch.cat([torch.arange(m, device=device)[None] < vis_lens[:, None],
                                   torch.ones((b, max_new + 1), dtype=torch.bool, device=device)], dim=1)

            for step in range(1, max_new):
                if not bool(unfinished.any()):
                    break
                xs = dec.token_embeddings(nxt.unsqueeze(1))
                cs, sn = cos_t[step].view(1, 1, -1), sin_t[step].view(1, 1, -1)
                smask = key_valid[:, None, None, :seqlen + 1]
                for i, layer in enumerate(dec.layers):
                    xs = layer(xs, mask=smask, cos_sin=(cs, sn), kv_cache=cache, layer_idx=i, cache_seqlens=seqlen)
                seqlen += 1
                nxt = dec.output_head(dec.final_norm(xs))[:, -1].argmax(-1).masked_fill(~unfinished, pad)
                out[:, step + 1] = nxt
                unfinished = unfinished & (nxt != eos) & (nxt != pad)
        return out.cpu()

    @torch.no_grad()
    def generate(self, pixel_values, pixel_attention_mask, spatial_shapes, tokenizer,
                 max_new_tokens: int = 128, num_beams: int = 1, repetition_penalty: float = 1.00,
                 dtype: torch.dtype = None, use_cuda_graph: bool = True, fast_engine: bool = None) -> list:
        """Greedy decoding (num_beams / repetition_penalty are accepted for API compat but ignored).

        Defaults need no configuration: fp16 + CUDA graphs on GPU, bf16 on CPUs with native bf16, fp32 on
        other CPUs; an enclosing torch.autocast dtype is honoured; `dtype=` overrides.
        `fast_engine=None` -> use the fused engine unless the weights are quantized (e.g. torchao int8),
        in which case the module-based path is used so the quantization is actually respected.
        Any non-OOM failure of the fast engine retries in fp32 and then uses the module-based path.
        """
        device = pixel_values.device
        if dtype is None:
            dtype = _default_dtype(device)

        bos_id = tokenizer.bos_token_id or 1
        eos_id = tokenizer.eos_token_id or 2
        pad_id = tokenizer.pad_token_id or eos_id

        if fast_engine is None:
            fast_engine = not _is_quantized(self)
        failed = self.__dict__.setdefault("_failed_dtypes", set())
        attempts = []
        if fast_engine:
            attempts.append((dtype, use_cuda_graph))
            if dtype != torch.float32:
                attempts.append((torch.float32, False))

        out = None
        for dt, want_graph in attempts:
            if dt in failed:
                continue
            try:
                with torch.inference_mode(), _no_autocast(device.type):
                    eng = self._engines.get(dt)
                    if eng is None or eng.device != device:
                        eng = self._engines[dt] = _FastEngine(self, dt)
                    graph = want_graph and device.type == "cuda" and max_new_tokens >= 4
                    out = eng.run(pixel_values, pixel_attention_mask, spatial_shapes,
                                  bos_id, eos_id, pad_id, max_new_tokens, graph)
                break
            except Exception as e:
                if _is_oom(e):
                    raise
                failed.add(dt)
                fr = traceback.extract_tb(e.__traceback__)[-1]
                warnings.warn(f"Fast generation path failed in {dt} ({type(e).__name__}: {str(e)[:200]}) "
                              f"at {os.path.basename(fr.filename)}:{fr.lineno}; falling back.")
        if out is None:
            out = self._reference_generate(pixel_values, pixel_attention_mask, spatial_shapes,
                                           bos_id, eos_id, pad_id, max_new_tokens, device)

        results = []
        for seq in out.tolist():
            clean = [t for t in seq[1:] if t not in (eos_id, pad_id)]
            results.append(tokenizer.decode(clean, skip_special_tokens=True))
        return results