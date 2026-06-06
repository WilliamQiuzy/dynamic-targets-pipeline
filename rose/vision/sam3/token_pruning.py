"""Fast-SAM2-inspired token pruning for SAM 3.1 multiplex memory attention,
with RoPE-aware position indexing so attention shapes stay consistent.

Status (2026-05-12)
-------------------
Structurally working but NOT a net win. SAM 3.1's encoder uses
``SimpleRoPEAttention`` (decoder.py) with rotary positional embeddings keyed
on the full 72×72 spatial grid. This patch makes RoPE pruning-aware so the
attention computation succeeds at K < 5184 query tokens.

Measured 2026-05-12 on Easy1/Easy2 at locked defaults (f=9):
- baseline:                4.76s ± 0.09s,  E1 (4,4,9)  E2 (5,5,9)
- token pruning dilate=4:  5.03s,  E1 (3,3,9)  E2 (2,2,9)   +5.6% slower, ⚠ quality
- token pruning dilate=2:  5.12s,  E1 (1,1,8)  E2 (2,2,9)   +7.5% slower, ✗ quality
- token pruning dilate=1:  4.97s,  E1 (1,1,8)  E2 (1,1,6)   +4.3% slower, ✗ quality

Two observations:
  1. **Encoder isn't the dominant cost** at f=9. The backbone (per-frame ViT
     feature extraction) is run once per frame with caching, but it still
     dominates wall time. The memory-attention encoder, even with O(K²) self
     attention, is small relative to backbone forward.
  2. **Pruning is out-of-distribution for a trained model.** Non-salient
     tokens get scattered back to their pre-encoder values. The mask decoder
     sees a discontinuous feature map (salient cells memory-conditioned,
     non-salient cells raw) and predicts inconsistent masks. Fast-SAM2 likely
     fine-tunes the model with their pruning policy; we can't do that here.

So this code stays as a research artifact. To make it a real speed win would
need: (a) compile/TensorRT the encoder to reduce wrapper overhead, AND
(b) fine-tune the SAM 3.1 weights with pruning active so the decoder learns
to handle the partial feature map. Without (b), quality drops even at
generous dilation.

This module patches ``SimpleRoPEAttention.forward`` to:
  - Gather ``freqs_cis`` for kept positions when pruning is active for q
  - For self-attention (q.shape[-2] == k.shape[-2]): use the gathered freqs
    for both q and k
  - For cross-attention to the memory bank (k.shape[-2] is a multiple of the
    full HW grid, since the memory bank is HW × num_memory_frames stacked):
    apply RoPE to q with gathered freqs and to k with the original full-grid
    freqs repeated per memory frame
  - Skip RoPE on the trailing ``num_k_exclude_rope`` object-pointer tokens
    just like the original

Plus a separate patch on ``_prepare_memory_conditioned_features`` and the
``TransformerEncoderDecoupledCrossAttention.forward`` that does the actual
gather/scatter around the encoder call.

Public API
----------
- ``install_token_pruning(predictor, stats=None)``
- ``uninstall_token_pruning(predictor)``
- ``build_per_frame_saliency_masks(anchor_masks, n_frames, feat_h, feat_w, dilate_cells=2)``
- ``set_saliency_on_model(predictor, masks)``
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ── Saliency mask construction ────────────────────────────────────────


def build_per_frame_saliency_masks(
    anchor_masks: Dict[int, torch.Tensor],
    n_frames: int,
    feat_h: int,
    feat_w: int,
    dilate_cells: int = 2,
) -> Dict[int, torch.Tensor]:
    """For each frame in [0, n_frames), build a feature-scale saliency mask."""
    if not anchor_masks:
        return {}
    anchor_indices = sorted(anchor_masks.keys())
    resized: Dict[int, torch.Tensor] = {}
    for idx, m in anchor_masks.items():
        if m.dim() == 2:
            m4 = m.float().unsqueeze(0).unsqueeze(0)
        else:
            m4 = m.float()
        m_small = F.interpolate(m4, size=(feat_h, feat_w), mode="nearest")
        resized[idx] = m_small.squeeze().bool()

    out: Dict[int, torch.Tensor] = {}
    for f in range(n_frames):
        before = [a for a in anchor_indices if a <= f]
        after = [a for a in anchor_indices if a >= f]
        chosen: List[int] = []
        if before:
            chosen.append(before[-1])
        if after and (not chosen or after[0] != chosen[-1]):
            chosen.append(after[0])
        mask = None
        for a in chosen:
            mask = resized[a] if mask is None else (mask | resized[a])
        if mask is None:
            continue
        if dilate_cells > 0:
            k = 2 * dilate_cells + 1
            dil = F.max_pool2d(
                mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=k, stride=1, padding=dilate_cells,
            )
            mask = dil.squeeze().bool()
        out[f] = mask
    return out


def set_saliency_on_model(predictor, masks: Optional[Dict[int, torch.Tensor]]):
    """Attach the per-frame saliency dict to the tracker.  ``None`` clears it."""
    predictor.model.tracker.model._frame_saliency_masks = masks


# ── Module-level state used by the patched RoPE forward ────────────────


# When non-None, the patched SimpleRoPEAttention.forward is told to gather
# freqs_cis at these indices for the q tokens. Module-level since the
# RoPE attentions live inside encoder layers we can't easily reach.
_active_keep_idx: Optional[Tensor] = None


def _complex_mult(xr, xi, fr, fi):
    return torch.stack([xr * fr - xi * fi, xr * fi + xi * fr], dim=-1)


def _reshape_freqs_for_broadcast(freqs: Tensor, x: Tensor) -> Tensor:
    """Broadcast freqs (N, D) over the head and batch dims of x (B, H, N, D)."""
    assert freqs.shape == (x.shape[-2], x.shape[-1]), \
        f"freqs {tuple(freqs.shape)} vs x_last2 {(x.shape[-2], x.shape[-1])}"
    return freqs.view(1, 1, x.shape[-2], x.shape[-1])


def _apply_rope_to_qk(
    q: Tensor, k: Tensor, freqs_q: Tensor, freqs_k: Tensor,
    use_rope_real: bool,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to q with freqs_q and to k with freqs_k (independent freqs).

    q: (B, H, N_q, D)   k: (B, H, N_k, D)
    freqs_q: (N_q, D/2) complex (or (N_q, D/2) for real); freqs_k similar.

    Returns rotated (q, k) at the same shapes.
    """
    qf = q.float().reshape(*q.shape[:-1], -1, 2)
    kf = k.float().reshape(*k.shape[:-1], -1, 2)
    qr, qi = qf[..., 0], qf[..., 1]
    kr, ki = kf[..., 0], kf[..., 1]

    if use_rope_real:
        fqr = _reshape_freqs_for_broadcast(freqs_q.real if torch.is_complex(freqs_q) else freqs_q, qr)
        fqi = _reshape_freqs_for_broadcast(freqs_q.imag if torch.is_complex(freqs_q) else freqs_q, qi)
        fkr = _reshape_freqs_for_broadcast(freqs_k.real if torch.is_complex(freqs_k) else freqs_k, kr)
        fki = _reshape_freqs_for_broadcast(freqs_k.imag if torch.is_complex(freqs_k) else freqs_k, ki)
    else:
        # complex tensors — split into real and imag
        fqr = _reshape_freqs_for_broadcast(freqs_q.real, qr)
        fqi = _reshape_freqs_for_broadcast(freqs_q.imag, qi)
        fkr = _reshape_freqs_for_broadcast(freqs_k.real, kr)
        fki = _reshape_freqs_for_broadcast(freqs_k.imag, ki)

    q_out = _complex_mult(qr, qi, fqr, fqi).flatten(-2)
    k_out = _complex_mult(kr, ki, fkr, fki).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


# ── Monkey-patch state ─────────────────────────────────────────────────


_ORIGINALS: Dict[str, callable] = {}


def install_token_pruning(
    predictor,
    min_keep_ratio: float = 0.15,
    max_keep_ratio: float = 0.85,
    stats: Optional[dict] = None,
):
    """Install three patches:
      1. ``Sam3MultiplexTracking._prepare_memory_conditioned_features`` — sets
         up an "active prune" plan when a saliency mask is present.
      2. ``TransformerEncoderDecoupledCrossAttention.forward`` — gathers
         queries before the encoder runs, scatters the encoder output back.
      3. ``SimpleRoPEAttention.forward`` — gathers ``freqs_cis`` so RoPE
         positions match the pruned query sequence.
    """
    tracker = predictor.model.tracker.model  # Sam3MultiplexTracking
    cls = type(tracker)
    enc = tracker.transformer.encoder
    enc_cls = type(enc)

    # Find the SimpleRoPEAttention class by importing — easier than walking
    from sam3.model.decoder import SimpleRoPEAttention

    if "_prepare" in _ORIGINALS:
        return  # already installed

    orig_prepare = cls._prepare_memory_conditioned_features
    orig_enc_fwd = enc_cls.forward
    orig_rope_fwd = SimpleRoPEAttention.forward
    _ORIGINALS["_prepare"] = orig_prepare
    _ORIGINALS["_encoder_forward"] = orig_enc_fwd
    _ORIGINALS["_rope_forward"] = orig_rope_fwd
    _ORIGINALS["_prepare_cls"] = cls
    _ORIGINALS["_enc_cls"] = enc_cls
    _ORIGINALS["_rope_cls"] = SimpleRoPEAttention

    # ── 1. _prepare_memory_conditioned_features ─────────────────────────
    def _patched_prepare(self, *args, **kwargs):
        if args:
            frame_idx = args[0]; is_init_cond_frame = args[1]
            current_vision_feats = args[2]
            current_vision_pos_embeds = args[3]
            feat_sizes = args[4]
        else:
            frame_idx = kwargs["frame_idx"]
            is_init_cond_frame = kwargs["is_init_cond_frame"]
            current_vision_feats = kwargs["current_vision_feats"]
            current_vision_pos_embeds = kwargs["current_vision_pos_embeds"]
            feat_sizes = kwargs["feat_sizes"]

        encoder = self.transformer.encoder
        encoder._active_prune = None

        saliency_dict = getattr(self, "_frame_saliency_masks", None)
        if (
            saliency_dict is not None and
            frame_idx in saliency_dict and
            not is_init_cond_frame and
            getattr(self, "num_maskmem", 0) > 0
        ):
            feat = current_vision_feats[-1]            # (HW, B, C)
            H, W = feat_sizes[-1]
            HW = H * W
            mask_hw = saliency_dict[frame_idx].to(feat.device)
            if mask_hw.shape != (H, W):
                mask_hw = F.interpolate(
                    mask_hw.float().unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest",
                ).bool().squeeze()
            mask_flat = mask_hw.flatten()
            K = int(mask_flat.sum().item())
            keep_ratio = K / HW
            if min_keep_ratio <= keep_ratio <= max_keep_ratio and K > 0:
                keep_idx = mask_flat.nonzero(as_tuple=True)[0]
                encoder._active_prune = {"keep_idx": keep_idx, "HW": HW}
                if stats is not None:
                    stats.setdefault("calls", 0)
                    stats.setdefault("keep_ratios", [])
                    stats["calls"] += 1
                    stats["keep_ratios"].append(keep_ratio)

        try:
            return orig_prepare(self, *args, **kwargs)
        finally:
            encoder._active_prune = None

    # ── 2. encoder forward — gather/scatter wrapper ─────────────────────
    def _patched_enc_forward(self, *args, **kwargs):
        plan = getattr(self, "_active_prune", None)
        if plan is None:
            return orig_enc_fwd(self, *args, **kwargs)

        keep_idx = plan["keep_idx"]

        def gather_token(t):
            if t is None:
                return None
            return t.index_select(0, keep_idx)

        feat_full_pre = kwargs.get("src", None)
        if feat_full_pre is None and args:
            feat_full_pre = args[0]
        if feat_full_pre is None:
            return orig_enc_fwd(self, *args, **kwargs)

        decoupled = ("memory" in kwargs) or ("memory_image" in kwargs)

        if decoupled:
            if "image" in kwargs:
                kwargs["image"] = gather_token(kwargs["image"])
            if "src" in kwargs:
                kwargs["src"] = gather_token(kwargs["src"])
            if "image_pos" in kwargs:
                kwargs["image_pos"] = gather_token(kwargs["image_pos"])
            if "src_pos" in kwargs:
                kwargs["src_pos"] = gather_token(kwargs["src_pos"])
        else:
            if args:
                args = list(args)
                args[0] = gather_token(args[0])
                if "src_pos" in kwargs:
                    kwargs["src_pos"] = gather_token(kwargs["src_pos"])
            else:
                kwargs["src"] = gather_token(kwargs["src"])
                if "src_pos" in kwargs:
                    kwargs["src_pos"] = gather_token(kwargs["src_pos"])

        # Activate RoPE patch for this call
        global _active_keep_idx
        _active_keep_idx = keep_idx
        try:
            out = orig_enc_fwd(self, *args, **kwargs)
        finally:
            _active_keep_idx = None

        # Scatter back (match dtype/device — encoder output may be fp32 after
        # our RoPE helper's float() round-trip).
        scatter = feat_full_pre.clone()
        scatter.index_copy_(0, keep_idx, out["memory"].to(scatter.dtype))
        out["memory"] = scatter
        return out

    # ── 3. SimpleRoPEAttention.forward — RoPE-aware gather ──────────────
    import math
    from sam3.model.decoder import functional_attention as _functional_attention

    def _patched_rope_forward(self, q, k, v, num_k_exclude_rope=0):
        keep_idx = _active_keep_idx
        # Move freqs to q device (mirrors original behaviour).
        self.freqs_cis = self.freqs_cis.to(q.device)

        # Fast-path: no pruning OR q is already at full size (e.g., this RoPE
        # block sees something else like a non-spatial sequence) → original.
        if (keep_idx is None) or (q.shape[-2] != int(keep_idx.numel())):
            return orig_rope_fwd(self, q, k, v, num_k_exclude_rope=num_k_exclude_rope)

        full_HW = self.freqs_cis.shape[0]
        if q.shape[-2] >= full_HW:
            # Shouldn't happen if our keep_idx is right, but safety net.
            return orig_rope_fwd(self, q, k, v, num_k_exclude_rope=num_k_exclude_rope)

        # ---- Projections + reshape to (B, H, N, head_dim) -----
        # SimpleRoPEAttention takes already-projected q/k/v; just reshape.
        b = q.shape[0]
        n_q = q.shape[1]
        n_k = k.shape[1]
        head_dim = q.shape[-1] // self.num_heads
        q_h = q.reshape(b, n_q, self.num_heads, head_dim).transpose(1, 2)
        k_h = k.reshape(b, n_k, self.num_heads, head_dim).transpose(1, 2)
        v_h = v.reshape(v.shape[0], n_k, self.num_heads, head_dim).transpose(1, 2)

        keep_idx_dev = keep_idx.to(q.device)
        freqs_full = self.freqs_cis  # (full_HW, D/2) complex (or whatever dtype)
        freqs_q = freqs_full.index_select(0, keep_idx_dev)  # (K, D/2)

        num_k_rope = n_k - num_k_exclude_rope

        if num_k_rope > 0:
            if num_k_rope == n_q:
                # Self-attention with both q and k pruned.
                freqs_k = freqs_q
            elif num_k_rope % full_HW == 0:
                # Cross-attention to memory bank (HW × num_mem_frames).
                r = num_k_rope // full_HW
                freqs_k = freqs_full.repeat(r, 1)
            else:
                # Doesn't match either pattern — bail to original.  This
                # restores freqs_cis (which the original may have mutated
                # under the hood) by passing the original q.
                return orig_rope_fwd(self, q, k, v, num_k_exclude_rope=num_k_exclude_rope)

            q_h, k_rope = _apply_rope_to_qk(
                q_h, k_h[:, :, :num_k_rope], freqs_q, freqs_k,
                use_rope_real=self.use_rope_real,
            )
            if num_k_exclude_rope > 0:
                k_h = torch.cat([k_rope, k_h[:, :, num_k_rope:]], dim=-2)
            else:
                k_h = k_rope

        # ---- Attention -----
        if self.use_fa3:
            from sam3.perflib.fa3 import flash_attn_func
            out = flash_attn_func(
                q_h.transpose(1, 2), k_h.transpose(1, 2), v_h.transpose(1, 2),
            ).transpose(1, 2)
        else:
            import torch.nn.functional as torchF
            out = torchF.scaled_dot_product_attention(q_h, k_h, v_h, dropout_p=0.0)

        out = out.transpose(1, 2).reshape(b, n_q, q.shape[-1])
        return out

    cls._prepare_memory_conditioned_features = _patched_prepare
    enc_cls.forward = _patched_enc_forward
    SimpleRoPEAttention.forward = _patched_rope_forward


def uninstall_token_pruning(predictor):
    if not _ORIGINALS:
        return
    _ORIGINALS["_prepare_cls"]._prepare_memory_conditioned_features = _ORIGINALS["_prepare"]
    _ORIGINALS["_enc_cls"].forward = _ORIGINALS["_encoder_forward"]
    _ORIGINALS["_rope_cls"].forward = _ORIGINALS["_rope_forward"]
    _ORIGINALS.clear()
