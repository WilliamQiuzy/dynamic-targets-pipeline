"""ROSE pipeline configuration dataclasses.

All hyperparameters are defined in the implementation spec
(docs/roadmap/ROSE_IMPLEMENTATION.md, Section 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# =============================================================================
# Vision Model Configs
# =============================================================================

@dataclass
class SAM3Config:
    """SAM3 segmentation / tracking configuration (Step 3, spec Section 6.1).

    model_path: local directory containing sam3.pt, or direct path to the
        checkpoint file.

    Note: SAM3 requires CUDA (Sam3VideoPredictor hardcodes .cuda()).
    SAM3 with text prompts always enables automatic instance detection
    (allow_new_detections is implicit in text-prompt mode).
    """
    score_threshold_detection: float = 0.3
    model_path: str = "rose/models/sam3"
    trim_past_non_cond_mem_for_eval: bool = True
    # 2026-05-31: OPTIMAL-CONFIG DEFAULTS. We are latency-bound; offloading SAM3
    # state/video to CPU adds per-frame CPU<->GPU transfers that hurt. H200 has
    # 141GB and we use only 14-31GB, so keep everything on-GPU. (Set True only if
    # running on a small-VRAM card.) Part of the validated 2.51 Hz prod config.
    offload_state_to_cpu: bool = False
    offload_video_to_cpu: bool = False
    max_discovery_per_frame: int = 2   # Max new objects via point prompt per frame (0=unlimited)
    max_discovery_total: int = 12      # Global cap on new objects across all frames (0=unlimited)
    discovery_min_mask_frac: float = 0.02  # Min mask area as fraction of image for discovery (skip tiny dets)
    enable_compile: bool = False       # Disabled for single-video runs: torch.compile recompiles
                                       # when object-count changes between phases (B-6 has 7 obj,
                                       # B-8 has 40 obj) and the compile/recompile overhead is NOT
                                       # amortised over a single video — measured 50s → 58s.
                                       # Re-enable only for batch processing (>10 videos) where
                                       # the cache warms up after the first 2-3 videos.
    chunk_size: int = -1   # -1=disable chunking (all frames in one pass, best quality), 0=auto, >0=explicit
    chunk_overlap: int = 3  # Overlap frames between adjacent chunks for mask IoU stitching
    # Inference acceleration (mechanism-level, no quality loss with default values)
    retain_backbone_cache: bool = True   # Keep backbone features across propagation passes (skip recompute in partial prop)
    num_maskmem: int = 7                 # Memory bank size for tracker. 2026-05-13: restored to SAM3 default
                                         # 7 (was 3 — that value was only safe at f=9 sampling; with dense
                                         # 10fps sampling we need the full memory horizon for long-range tracking).
    memory_temporal_stride: int = 4      # 2026-05-13: restored from 8 → 4 for dense sampling. Keeps a 28-frame
                                         # memory horizon at num_maskmem=7.
    # ---- Re-tracking knobs (SAM 3.1's native memory bank) -----------
    # SAM 3.1's tracker has two attention sets:
    #   • conditioning frames (any frame with a prompt) — addressed by max_cond_frames_in_attn
    #   • non-conditioning memory — addressed by num_maskmem × memory_temporal_stride
    # The multiplex checkpoint ships with max_cond_frames_in_attn=4 and
    # keep_first_cond_frame=False, which means once we add a few refines the
    # original frame-0 bbox prompt can be evicted from attention — that kills
    # long-horizon re-acquisition.  We unlock both so SAM 3.1's native re-
    # tracking has full memory at its disposal.
    max_cond_frames_in_attn: int = -1    # -1 = unlimited (let SAM 3.1 attend to every conditioning frame)
    keep_first_cond_frame: bool = True   # Anchor the initial prompt frame — never evict it
    # SAM 3.1's NATIVE memory frame selection.  When True, the tracker uses
    # ``frame_filter`` to pick past frames whose mask predictions had high
    # effective IoU score (``eff_iou_score > mf_threshold``) for memory
    # attention — skipping frames where the object was lost.  Stronger
    # re-tracking after short occlusions than the default "last-N consecutive
    # frames" policy.  Comes from SAM 3.1's own multiplex code (off by default).
    use_memory_selection: bool = True
    mf_threshold: float = 0.01           # Min eff_iou_score for a past frame to enter memory
    # Phase B-8 strategy: how to handle late-anchor discoveries that didn't
    # match a session-1 track in Phase B-7.5.
    #   "session_reset"  (legacy) — open a fresh multiplex session 2, register
    #                    each late candidate as a point prompt, propagate.
    #                    Observed pathology: session-2 propagation is sparse —
    #                    masks emit ONLY at anchor frames, not in between.
    #   "in_session"     — keep session 1 active, add each late candidate as
    #                    a NEW obj_id via add_object_point at its anchor
    #                    frame, re-propagate session 1.  All objects share
    #                    one memory bank → continuous propagation.
    late_discovery_mode: str = "in_session"  # 2026-05-13: restored from "off". Dense 10fps sampling means more
                                         # frames where new objects can appear; B-8 in_session is needed for
                                         # mid-video discoveries. (Was "off" only as a speed shortcut at f=9.)
    parallel_lift: bool = True   # 2026-05-31 DEFAULT: parallelise the per-frame 4DSG backprojection
                                 # (compute_candidates) across CPU cores. It is the dominant 4DSG cost
                                 # (~90% of build_4dsg, ~30% of e2e), runs at full 10fps (un-strideable),
                                 # and is PURE per frame → numerically identical to serial. The heavy
                                 # numpy (nonzero/gather/matmul/percentile/patch-tokens) releases the GIL.
    parallel_lift_workers: int = 16  # thread pool size for parallel_lift (box has 192 cores).
    early_dedup_iom: float = 0.7  # 2026-05-31 DEFAULT (recommended speed config): Phase B-7.6 early dedup. >0 drops late candidates
                                  # whose discovery-frame mask has IoM (intersection/min-area) >= this
                                  # with an existing track at that same frame (spatial duplicate →
                                  # already tracked → skip its B-8 propagation). 0.7 = quality-neutral
                                  # B-8 object-count cut. 0.0 = off.
    incremental_late_prop: bool = True  # 2026-05-31 DEFAULT (recommended speed config): B-8 re-propagates ALL frames for late-discovered
                                     # objects (the 73%% bottleneck).  A late object first appears at
                                     # frame f, so frames [0,f) need not be re-propagated.  When True, B-8
                                     # propagates only from the earliest late-discovery frame onward,
                                     # preserving earlier cached masks.  Launch-reducing (helps our
                                     # latency-bound regime, unlike memory-retrieval/num_maskmem which
                                     # only cut FLOPs and gave 0%% speedup).
    flow_warp_skipped: bool = True   # 2026-05-31: Tier-4. With sam3_propagate_stride>1, skipped frames
                                     # get masks COPIED from the nearest keyframe (stair-step in 3D).
                                     # When True, instead WARP the keyframe mask to the skipped frame via
                                     # dense optical flow (Farneback) → motion-compensated mask → smooth
                                     # trajectory.  Flow is ~1s/video << the SAM3 frames saved by striding.
    interleaved_dual_stream: bool = False  # 2026-05-30: GPU-async full-quality speedup. Propagate EVEN
                                     # frames on a 2nd SAM3 instance / CUDA stream and ODD frames on the
                                     # main one CONCURRENTLY; the two fill each other's ~50%% GPU idle so
                                     # all frames get a REAL mask (no copy → no centroid stair-step) at
                                     # ~2x.  Costs a 2nd SAM3.1 in VRAM.  Each half tracks at 5fps memory.
    sam3_propagate_stride: int = 3   # 2026-05-31 DEFAULT (recommended speed config, was 2): TEMPORAL SUBSAMPLING. 1=propagate every frame.
                                     # >1 propagates SAM3 on every Nth frame (e.g. 2 = 5fps) and copies
                                     # each mask to the (N-1) skipped frames; DA3 + 4DSG stay at full
                                     # 10fps.  Propagation cost (the 73%% B-8 bottleneck) scales ~1/N.
                                     # Object DISCOVERY still uses every frame given to _run_phase2, so
                                     # striding the whole phase reduces anchor density — validate object
                                     # count when raising.
    full_propagation_stride: int = 5     # Discovery frame stride: FastSAM runs every N-th frame during Phase 2b (1=all, 5=every 5th)
    vg_stride: int = 25                  # 2026-05-13: restored from 0 → 25. VG re-grounding every 25 frames
                                         # helps catch objects that drift / get occluded mid-video.
    max_init_masks: int = 20              # 2026-05-13: restored from 8 → 20. Dense sampling reveals more
                                          # objects per frame; let SAM 3 keep more candidates.
    max_active_tracks: int = 32           # 2026-05-30: raised 16 → 32 to free the seeding-dedup benefit:
                                          # the containment fix yields ~60 DISTINCT object seeds, but a 16-cap
                                          # threw most away (kept top-16 by area).  32 captures markedly more
                                          # real objects (clip_001 10→14 tracks) with NO extra fragmentation,
                                          # at ~25% more SAM3 time (2 multiplex buckets instead of 1).  Set back
                                          # to 16 for max speed.  2026-05-13: was restored from 8 → 16.
                                          # Dense sampling can surface more concurrent objects.
    # Flash-Attention master switch (warm_server applies it at load).
    #   enable_fa3 = False (DEFAULT): force every attention module to use_fa3=False
    #     → PyTorch SDPA picks the best kernel per GPU (flash on A100+/Hopper,
    #       mem-efficient/math on older cards like V100). Runs on ANY CUDA GPU.
    #   enable_fa3 = True: Hopper (H100/H200) ONLY, with FA3 compiled in — max speed.
    # Env override: ROSE_DISABLE_FA3=1 forces it off regardless.
    enable_fa3: bool = False
    use_fa3: bool = True                  # 2026-05-31: OPTIMAL default (part of validated 2.51 Hz prod config).
                                          # Enables FA3 on SAM3's intended long-seq attention mix (NOT fa3_everywhere,
                                          # which flips the short-seq modules too and is ~3% SLOWER — keep that False).
                                          # Requires `flash_attn_interface` (built from Dao-AILab/flash-attention/hopper,
                                          # already compiled for sm_90 on this H200 pod + baked into the Docker image).
                                          # Patches use_fa3=True post-load on the RoPE/Sam/ViT attention modules.
    use_cutie_propagation: bool = False  # 2026-05-31: replace SAM3 per-frame tracking (the 73%% B-8
                                     # bottleneck) with Cutie (pretrained real-time VOS, ~6-18fps).
                                     # FastSAM+dedup discovers objects; Cutie propagates their masks
                                     # across all frames (mid-video object adds supported).  ~3.7-4.3x
                                     # faster propagation, no training. Needs the `cutie` package.
    use_multiplex: bool = True             # SAM 3.1 Object Multiplex predictor (joint multi-object
                                          # tracking, ~7x faster at high object counts).  This is the
                                          # QUALITY path: joint tracking avoids the per-object session
                                          # competition that makes the base SAM3 path alias one physical
                                          # object across several unstable tracks (e.g. a paint tray
                                          # fragmenting into 4-5 tracks).  Needs sam3.1_multiplex.pt at
                                          # multiplex_model_path (downloaded from facebook/sam3.1).
                                          # Set False to fall back to base facebook/sam3 if the SAM3.1
                                          # checkpoint is absent.
    multiplex_model_path: str = "rose/models/sam3.1"  # Directory containing sam3.1_multiplex.pt
    multiplex_count: int = 16              # Bucket size for multiplex (must be >= max_active_tracks)
    late_match_temporal_window: int = 2    # ±frames window in which a late-anchor FastSAM detection is
                                           # matched back to an existing SAM3 track (re-link instead of new id).
                                           # If an object is lost for MORE than this many frames before being
                                           # re-detected, the re-link fails and a NEW id is allocated →
                                           # fragmentation.  Widen to re-link across longer gaps.
    late_match_use_dino: bool = False      # 2026-05-31: upgrade B-7.5 Tier-3 (visual) matcher from HSV color
                                          # histogram → DINOv2 embedding (same descriptor the post-hoc reid_merge
                                          # uses with 0 false merges). Catches re-appearance duplicates EARLIER
                                          # (before B-8 propagates them), cutting B-8 object count → faster, while
                                          # folding the re-appearance into the existing track → observations
                                          # PRESERVED (quality-safe, unlike stride/incremental which drop obs).
    late_match_dino_thresh: float = 0.62   # DINOv2 cosine for the strong-alone Tier-3 gate (proximity gate uses
                                          # thresh-0.08). Conservative/high-precision to avoid false early-links.
    gpu_late_match: bool = True            # 2026-06-01: GPU-batch the B-7.5 mask-IoU matching. The CPU matcher
                                          # recomputes np.where centroids per-candidate and does full-res
                                          # logical_and/or per (cand,frame,mask). We precompute m1 area/centroid
                                          # ONCE and compute the intersection via a float32 matmul of 0/1 masks
                                          # (exact integer ≤ HW ≪ 2**24). union = |A|+|B|-|A∩B| (set identity).
                                          # → BIT-IDENTICAL decisions (same int IoU, same thresholds). Set False
                                          # to fall back to the original per-candidate CPU loop (A/B reference).
    gpu_anchor_dedup: bool = True          # 2026-06-01: GPU-batch the B-2 cross-anchor mask-IoM matrix
                                          # (FastSAM 30→81 dets made B-2's O(M²) CPU IoM 0.3→1.7s). Only the
                                          # exact-integer INTERSECTION is done on GPU (float32 matmul of 0/1
                                          # masks, exact ≤ HW); the IoM ratio + 0.7 threshold compare stay in
                                          # float64 Python identical to the CPU path → BIT-IDENTICAL unique set
                                          # (no float32 boundary flips). Set False for the pure-CPU reference.
    skip_b4_frame0_reground: bool = True   # 2026-06-01: B-3 (add_bboxes_batch) already grounds the
                                          # prompt frame (~0.6s detection) and sets the model's
                                          # cached_frame_outputs[prompt_frame]; B-4's full-VG propagation then
                                          # RE-grounds that same frame (~0.6s) before tracking 1..N (~0.05s each).
                                          # When True, B-4 starts at prompt_frame+1, reusing B-3's detection →
                                          # saves ~0.6s/video IF the cost is real detection (not first-frame-of-
                                          # pass overhead that just moves to frame+1). B-6 re-propagates [0..N]
                                          # afterwards so the mask cache is unaffected. A/B verified 2026-06-01:
                                          # saves ~0.34s mean (0.65-0.74s crowded), quality within noise → default ON.
    skip_b6_redetect: bool = True          # 2026-06-01: B-6's SAM2-partial propagation calls
                                          # _prepare_backbone_feats per frame → run_backbone_and_detection,
                                          # which RE-RUNS the full FA detection (~1.5s/video) only to recover
                                          # backbone features that B-3/B-4 already computed but the per-frame
                                          # feature_cache EVICTED (sliding-window, sam3_multiplex_base.py:794).
                                          # When True: (a) don't evict per-frame backbone features (H200 has
                                          # ~110GB free, ~17 strided frames is cheap), (b) _prepare_backbone_feats
                                          # reuses the cached features instead of re-detecting. Saves ~1.5s/video,
                                          # quality-neutral in principle (same backbone feats). A/B verified
                                          # 2026-06-01: ~2.39s/video faster (~23%, median ~4.0→~5.2 Hz),
                                          # track/obs within noise, GPU peak unchanged → default ON.
    flow_warp_workers: int = 16            # 2026-06-02: _flow_warp_skipped runs cv2 Farneback optical flow
                                          # per SKIPPED frame, all independent. ThreadPool the per-frame flow
                                          # (cv2 releases the GIL) on the 192-core box. A/B verified: flow_warp
                                          # 0.86s→0.09s (8.6x), e2e −0.77s/video, quality identical (numerically
                                          # same algorithm) → default 16. Set 1 for serial reference.
    defer_session_cleanup: bool = True     # 2026-06-02: _cleanup_sam3_session (end_all_runs →
                                          # close_session, frees session GPU tensors) runs in the finally
                                          # of _run_inference_impl, BLOCKING the caller's return by ~0.46s.
                                          # The 4DSG result is already built (masks copied to CPU) before
                                          # the finally, so the session tensors are safe to free afterwards.
                                          # When True: run cleanup in a daemon thread and JOIN it at the
                                          # start of the next inference (before a new session opens) →
                                          # ~0.46s off single-video latency, quality-neutral. A/B verified
                                          # 2026-06-02: ~0.5-0.65s/video faster, quality BYTE-IDENTICAL
                                          # (1546/1546 obs) → default ON. NOTE: real single-video latency win
                                          # (request→response sooner); a back-to-back benchmark loop hides it
                                          # (next video joins prev cleanup at its start) — but never regresses.
    anchor_stride: int = 4                 # 2026-05-13: restored from 8 → 4. With dense 10fps sampling we
                                          # want more FastSAM anchor passes to catch objects appearing later.
    # ---- Fast-SAM2-style token pruning (arxiv 2512.21333) ----
    compile_dynamic: bool = True   # 2026-05-30: torch.compile dynamic flag. dynamic=True
                                 # avoids per-shape recompiles but limits CUDA-graph capture
                                 # (eager fallback on varying shapes). dynamic=False forces
                                 # STATIC shapes → fuller CUDA-graph capture → faster, at the
                                 # cost of a recompile per distinct shape at warmup.
    compile_memory_encoder: bool = True   # torch.compile the SAM 3.1 multiplex memory-attention encoder
                                          # (TransformerEncoderDecoupledCrossAttention.forward). Steady-state
                                          # ~4% speedup on Easy1/Easy2 (4.66 → 4.48s), no quality regression.
                                          # First inference call takes ~3-4 min to JIT-compile.
    compile_tracker_strong: bool = False   # 2026-05-31: EXPERIMENTAL (user's "compile SAM2 modules separately").
                                          # The full native SAM3.1 _compile_model ABORTS on torch 2.4 — but the
                                          # crash is in backbone.vision_backbone.trunk (the SAM3 DETECTOR), NOT
                                          # the SAM2-derived tracker. So compile ONLY the tracker's 3 SAM2 modules
                                          # (maskmem_backbone, transformer.encoder=memory-attn, sam_mask_decoder)
                                          # with the native STRONG recipe (fullgraph max-autotune; encoder uses
                                          # max-autotune-no-cudagraphs+dynamic=True for the growing memory bank,
                                          # the fixed-shape decoder/maskmem get max-autotune+dynamic=False → real
                                          # CUDA-graph capture). Skips the detector that crashes. Takes priority
                                          # over compile_memory_encoder/compile_mask_decoder_transformer. bit-exact.
    cuda_graph_memory_encoder: bool = False  # 2026-05-31: EXPERIMENTAL. Replace the torch.compile wrap of the
                                          # memory-attention encoder with a MANUAL shape-keyed CUDA graph
                                          # (cuda_graph_module.CUDAGraphedModule). torch.compile(dynamic=True)
                                          # DISABLES cudagraphs (only fuses → ~4%); manual capture replays the
                                          # whole encoder.forward per steady-state shape, killing kernel-launch
                                          # overhead. bit-exact, bf16-safe, falls back to eager on any failure.
                                          # Mutually exclusive with compile_memory_encoder (this takes priority).
    fa3_everywhere: bool = False         # 2026-05-13: tried walking the tree and flipping all 20 use_fa3=False
                                          # attentions to True. Measured +3.4% slower (4.72→4.88s ± 0.13s).  FA3
                                          # uses FP8 internally; the FP8 cast overhead only pays off for long
                                          # sequences (backbone HW=5184, encoder HW=5184). The 20 default-False
                                          # modules are exactly the short-seqlen ones (mask decoders with ~12-32
                                          # query tokens, detector.geometry_encoder); SDPA-flash (FA2 in bf16) is
                                          # faster there. Model authors had the right default. Keep this off.
    compile_mask_decoder_transformer: bool = True  # 2026-05-13: compile the mask-decoder's TwoWayTransformer.
                                          # With the FA3 patches now in model_misc.py (qkv-same-embed + no-op mask),
                                          # this finally pays off: 4.53 → 4.27s (-5.7%) on top of encoder.compile,
                                          # 6-run stdev 0.044s, quality fully preserved. The use_fa3=True bypass
                                          # forces FA3 kernels which AOTAutograd treats as opaque, so SDPA
                                          # decomposition (which previously broke mask logits) doesn't happen.
    use_token_pruning: bool = False        # Fast-SAM2-style token pruning prototype (rose/vision/sam3/token_pruning.py).
                                          # KNOWN BROKEN on SAM 3.1 multiplex: the encoder's RoPE attention has
                                          # fixed feat_sizes=[72,72], so pruning tokens to a smaller sequence breaks
                                          # the spatial position embeddings (shape mismatch 5184 vs K_kept at the
                                          # attention layer). Speed potential is real (~60% drop when pruning runs,
                                          # measured 2026-05-12), but quality dies (0 tracks). A correct
                                          # implementation needs RoPE position remapping. Kept here for reference.
    token_prune_feat_size: int = 72       # Feature-grid resolution at which we build the saliency mask. SAM 3.1
                                          # multiplex's encoder runs at 72×72 (image_size=1008, stride=14).
    token_prune_dilate_cells: int = 2     # Dilation kernel half-width in feature cells (absorbs object motion).
    cross_anchor_iou_thresh: float = 0.4   # Cross-anchor bbox IoU threshold for dedup (lower = more aggressive merge)
    seed_containment_iom: float = 0.7      # Mask intersection-over-min gate for pre-seeding dedup (0 disables).
                                           # ROOT-CAUSE FIX for fragmentation: FastSAM proposes one physical object
                                           # as several overlapping/NESTED part-masks (e.g. a tray + the paint blob
                                           # inside it).  bbox-IoU dedup misses nesting (their bbox-IoU < 0.4), so
                                           # each part was seeded as a SEPARATE SAM3 object → SAM3.1 faithfully tracks
                                           # all of them → concurrent "fragments" with ~0 mutual mask-IoU.  IoM ≥ 0.7
                                           # means the smaller mask is ≥70% inside the larger → same object → keep one.
    discovery_backward_window: int = 10   # Bounded backward propagation window (D4): an object discovered at
                                          # frame F triggers re-propagation starting at max(0, F - window) instead
                                          # of frame 0.  Mini-200 ablation: window=10 gives +13% Hz with n_obs/
                                          # n_tracks fully preserved.  window=0 saves more time but drops obs ~7%
                                          # (objects lose their early-frame trajectory).  Set to >>num_frames to
                                          # disable bounding entirely.


@dataclass
class DA3Config:
    """DA3 monocular depth estimation configuration (Step 2).

    model_path: local directory or HF hub ID passed to
        ``DepthAnything3.from_pretrained()``.

    Supported model variants (smallest → largest):
      - ``da3-small``   (~34M)  — relative depth + K + T_wc
      - ``da3-base``    (~120M) — relative depth + K + T_wc
      - ``da3-large``   (~350M) — relative depth + K + T_wc
      - ``da3nested-giant-large`` (~1.4B) — metric depth + K + T_wc

    When ``require_metric`` is False (default), relative-depth models are
    accepted and 3D coordinates use an arbitrary but consistent scale.
    Set ``require_metric=True`` to enforce metric depth in metres.

    Chunked inference (see docs/bugs/DA3_BATCH_OOM.md):
      ``infer_batch`` automatically splits frames into overlapping chunks
      and aligns them via SIM3 point-cloud matching when the batch would
      exceed the chunk limit.

      chunk_size semantics:
        - ``0``  (default): **auto** — estimate safe chunk size from free
          GPU memory.  Prevents OOM on large videos without manual tuning.
        - ``> 0``: explicit max frames per chunk.
        - ``< 0``: disable chunking (always single forward pass; may OOM).

      If a single-batch forward pass triggers OOM, the wrapper
      automatically falls back to chunked inference at half the batch size.
    """
    device: str = "cuda"
    model_path: str = "rose/models/da3nested-giant-large"  # 2026-05-30: switched small/large →
                                          # giant-nested for METRIC depth.  The 4DSG's whole value is its
                                          # 3D coordinates; relative-depth models gave ambiguous scale
                                          # (a skier's extent read 0.09 "units").  Giant gives real metres
                                          # and is nearly free (DA3 is ~2.6% of pipeline time).  Robust
                                          # percentile extents (geometry_tokens.build_shape_token) tame the
                                          # far-object depth-noise inflation that metric depth otherwise shows.
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    require_metric: bool = True           # 2026-05-30: enforce metric depth (giant-nested IS metric)
    chunk_size: int = -1   # -1 = no chunking (single forward pass for all 32 frames).
                           # H200 has 141 GB and can fit our typical 32-frame batch
                           # easily; eliminating chunking saves ~1-2s of SIM3 alignment
                           # + chunk-boundary overhead.  Auto-fallback to chunked on OOM.
    chunk_overlap: int = 5  # Overlap frames between adjacent chunks for SIM3 alignment.


@dataclass
class RAMPlusConfig:
    """RAM++ image tagging configuration (Step 1)."""
    device: str = "cuda"
    model_path: str = "rose/models/ram_plus"
    checkpoint_path: Optional[str] = None  # Explicit .pth path; auto-detect if None
    normalize_lowercase: bool = True
    deduplicate_tags: bool = True


@dataclass
class YOLOConfig:
    """YOLO bbox detection configuration (legacy, used by asset scripts)."""
    device: str = "cuda"
    model_path: str = "yolo11n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    imgsz: int = 640
    max_det: int = 200


@dataclass
class FastSAMConfig:
    """FastSAM class-agnostic segmentation configuration (Step 1).

    Replaces YOLO for initial object discovery.  FastSAM produces
    instance masks without class labels, enabling open-world detection.
    """
    device: str = "cuda"
    model_path: str = "rose/models/fastsam/FastSAM-s.pt"
    conf_threshold: float = 0.55
    iou_threshold: float = 0.9
    imgsz: int = 640
    max_det: int = 200
    discovery_iou_thresh: float = 0.3  # Below this IoU = new object in two-pass discovery
    max_mask_frac: float = 0.35  # Skip masks covering > this fraction of the image (0=disabled).
                                 # 0.35 keeps medium-size objects (boat ~17% of frame, hill
                                 # village ~3%, car ~25%) but excludes blob masks that cover
                                 # half the frame (which lose object identity by combining
                                 # multiple structures into one).  Late-discovery in B-8 picks
                                 # up larger foreground subjects via the second multiplex
                                 # session even when this filter excludes them at frame 0.


@dataclass
class SamplingConfig:
    """Frame sampling and scheduling configuration (Step 0)."""
    target_fps: float = 10.0
    max_frames: Optional[int] = None     # 2026-05-13: removed the cap. The whole 4DSG pipeline runs at
                                         # `target_fps` × video duration frames. Set to an int to cap; None
                                         # honours target_fps throughout the video. The earlier f=9 cap was
                                         # a speed shortcut that's no longer the policy — 4DSG quality wants
                                         # genuine 10 fps temporal coverage.


# =============================================================================
# Pipeline Step Configs
# =============================================================================

@dataclass
class DepthFilterConfig:
    """Depth & 3D filtering configuration (Step 4, spec Section 6.2)."""
    conf_thresh: float = 0.5
    min_points: int = 50
    max_extent: float = 30.0  # meters


@dataclass
class FusionConfig:
    """Global ID fusion configuration (Step 5, spec Section 6.3)."""
    cross_run_iou_thresh: float = 0.4   # Lowered from 0.75 — same physical object tracked by two
                                        # different SAM3 sessions in the same frame typically only
                                        # overlaps 0.4-0.6 (pose differences, mask edge noise).
                                        # 0.4 catches those duplicates while still keeping spatially
                                        # adjacent-but-distinct objects separate via the centroid gate.
    merge_centroid_dist_m: float = 2.0
    merge_centroid_dist_rel: float = 0.2  # Depth-normalised distance gate for relative depth (cdist / mean_z)
    merge_temporal_gap: int = 10        # Raised from 2 — SAM3 anchors are 4 frames apart; the same
                                        # object is routinely picked up by sessions whose last_seen_t
                                        # differs by >>2 frames.  10 covers a typical 32-frame chunk.
    lost_patience: int = 5   # frames
    archive_patience: int = 30  # frames
    # Post-hoc track deduplication (merge re-tracked objects)
    enable_post_dedup: bool = True   # Master switch for merge_duplicate_tracks.  With in_session
                                     # late-discovery producing DENSE multi-track propagation per
                                     # physical object, post-pipeline dedup is the right place to
                                     # merge identity duplicates: same horse tracked twice → near-
                                     # identical 2D image trajectories → safe to merge.
    dedup_crop_sim_thresh: float = 0.94  # Crop visual similarity gate (used in first pass of
                                         # merge_duplicate_tracks).  Tight to avoid merging
                                         # distinct objects that share rough color palette.
    traj_merge_dist: float = 0.10  # 2D image-frame bbox-center distance gate, paired with
                                   # crop-similarity ≥ 0.65 (AND-gate inside merge_duplicate_tracks
                                   # pass 2).  10% of image diagonal accommodates the small bbox
                                   # offset between part-level masks of the same physical object
                                   # without merging unrelated tracks that happen to share image
                                   # space (e.g. batter swinging + ground patch behind them).
    mask_traj_iou_thresh: float = 0.25  # PRIMARY duplicate-track filter: mean mask-IoU between
                                       # two tracks across shared frames.  With in_session late-
                                       # discovery, dupes of the same physical object now share
                                       # 20+ frames densely, so 0.25 is safe.  The dedup also
                                       # gates on min-shared-frames (4) + area ratio (0.25) to
                                       # avoid merging brief image-space crossings of distinct
                                       # objects.  Operates on full masks so it's immune to DA3
                                       # relative-depth issues that broke 3D-centroid dedup.
    # ── Deep re-ID track merge (fix for same-object fragmentation that mask-IoU
    #    misses: e.g. a paint tray tracked by 2-5 concurrent obj_ids whose masks
    #    land on ALTERNATING frames → ~0 mutual IoU, but their best-crops are the
    #    same object).  Uses a DINOv2 embedding of each track's best crop —
    #    validated far more discriminative than raw-pixel cosine (which produced
    #    2-9 false merges on clip_001; DINOv2 single-crop+gate → 0).  Catches both
    #    concurrent-alternating dups and disappear→reappear re-tracks.
    reid_merge: bool = True
    reid_sim_thresh: float = 0.72     # cosine ≥ this on DINOv2 best-crop embeddings → merge.
                                      # Validated stable on clip_001 over [0.68, 0.78]: 2/2 true
                                      # dups merged, 0 false merges.
    reid_model: str = "vit_base_patch14_dinov2.lvd142m"  # timm model for the embedding
    reid_max_2d_jump: float = 0.25    # for CONCURRENT tracks, require mean 2D-center distance ≤ this
    reid_max_gap_s: float = 0.6       # for DISJOINT tracks (disappear→reappear), max time gap
    # ── Sequential re-link (fix for disjoint "断帧": object lost then re-tracked
    #    under a new id with NO overlapping frames, so mask-IoU dedup can't see
    #    it).  A new track is re-linked to a just-ended one ONLY when ALL gates
    #    hold, which rejects the blurry/low-texture false positives that crop-
    #    similarity alone produces (e.g. uniform outdoor scenes).
    enable_seq_relink: bool = True
    seq_relink_max_gap_s: float = 0.6    # max time between A's last and B's first obs
    seq_relink_max_3d_jump: float = 0.2  # 3D centroid continuity at the break (world units);
                                         # the discriminative gate — distinct objects jump >>this
    seq_relink_max_2d_jump: float = 0.2  # normalized image-center continuity at the break
    seq_relink_crop_sim: float = 0.88    # appearance confirmation (cosine sim of crops)
    min_track_observations: int = 3  # Drop tracks with fewer observations: ghost tracks where
                                     # SAM3 produced 1-2 stray masks before losing the object.
                                     # 3 is the sweet spot — real long-term tracks have n≥10,
                                     # so this kills only ghost / partial tracks.
    max_track_extent: float = 15.0   # 2026-05-30: recalibrated 1.3 → 15.0 for METRIC depth (giant).
                                     # In metres, 15 m on any axis means the "object" is a background /
                                     # scene region (sky, far trees, a whole building wall), not a
                                     # discrete trackable object.  Discrete foreground subjects — even a
                                     # close-up dog (~1 m) or a slalom banner (~5 m) — are well under it
                                     # and kept.  (Old 1.3 was a RELATIVE-depth normalized-units value and
                                     # would drop nearly everything once depth is in real metres.)


@dataclass
class STEPConfig:
    """STEP token configuration (Step 6, spec Section 6.4)."""
    grid_size: int = 16
    iou_threshold: float = 0.5
    mask_outside_pixels: bool = True  # Zero out non-mask pixels in patch crops (paper default)
    patch_crop_size: int = 64  # Resize each cell crop to 64x64 for uniform visual tokens
    temporal_window: int = 0  # F_k sliding window: 0 = keep all observations (no truncation)
    max_tau_per_step: int = 0  # Top-k patches per STEP token, sorted by IoU desc. 0=unlimited.


@dataclass
class VLMConfig:
    """VLM inference configuration (Step 9).

    Supports two providers:
    - "openai": Uses OpenAI API (default, for GPT-5.2 etc.)
    - "google": Uses Google genai API (for Gemini, Gemma etc.)

    The API key is read from the environment variable specified by api_key_env.
    """
    provider: str = "openai"          # "openai" | "google"
    model: str = "gpt-5.2"            # Model name sent to API
    max_output_tokens: int = 1024
    temperature: float = 1.0
    api_key_env: str = "OPENAI_API_KEY"  # Env var name for the API key
    base_url: Optional[str] = None       # Optional base URL override (e.g. for Gemini via OpenAI-compat)
    object_crop_size: int = 256          # Per-object masked crop resize target (px)
    object_crop_padding: float = 0.2     # Bbox padding ratio for object crops


@dataclass
class DynamicTargetsConfig:
    """Dynamic-targets export (动态目标管线): per-frame ALL_FRAMES JSON with
    9DOF oriented 3D boxes + instance names + 2D boxes + absolute velocity.

    This is an ADDITIVE output module. When ``enabled`` is False (default) the
    pipeline is byte-for-byte the validated 5.56 Hz path — none of the extra
    geometry (OBB) or VLM naming runs. Set ``enabled=True`` to emit the format.
    """
    enabled: bool = False                # Master gate. OFF = 5.56 default path untouched.
    # Instance naming (open-vocabulary, local multimodal VLM).
    # Bake-off (close-up COCO + ROSE crops, 2026-06-04): Qwen3-VL-4B best (73.8%),
    # then Qwen2.5-VL-7B (72.5%), Gemma-3-4B (65%), DeepSeek-VL-7B (56%).
    namer_provider: str = "qwen"         # advisory; namer_model_path selects the model.
    namer_model_path: str = "rose/models/qwen3-vl-4b-instruct"   # DEFAULT = bake-off winner.
    qwen_model_path: str = "rose/models/qwen3-vl-4b-instruct"    # (back-compat alias)
    gemma_model_path: str = "rose/models/gemma-3-4b-it"
    name_objects: bool = True            # Run the VLM namer (else instance_name="object").
    max_crops_per_object: int = 3        # Top-K sharpest crops (distinct frames) used per object.
    namer_vote: bool = True              # True: name each of the K crops + majority vote (robust to a
                                         # blurry/partial view); False: single best-crop call (~3x faster).
    namer_max_new_tokens: int = 16
    # 9DOF oriented box.
    obb_extent_pct: float = 2.0          # Robust trim percentile for per-frame box bounds.
    obb_min_points: int = 12             # Below this, fall back to axis-aligned / skip box.
    stabilize_size: bool = True          # Use track-median box size (rigid object) + per-frame
                                         # center & yaw → reduces per-frame monocular jitter.
    # Velocity + center estimation. Default = constant-velocity RTS Kalman smoother
    # with innovation gating + depth-adaptive measurement noise. On a synthetic
    # benchmark (static/const/turning trajectories, depth-scaled noise + 8% bad-depth
    # outlier frames) this cut velocity RMSE 0.99→0.27 m/s (3.7x) vs the old
    # moving-average + windowed-least-squares method, and also improved center RMSE.
    # The gate auto-rejects bad-depth frames that spike velocity; R∝depth^2
    # down-weights noisy far measurements. Center reported = Kalman smoothed position;
    # velocity reported = Kalman velocity state.
    velocity_kalman: bool = True         # False → legacy MA-smooth + windowed-LSQ.
    kalman_sigma_a: float = 0.5          # process accel noise (m/s^2); lower = smoother. 0.5 tuned on the
                                         # drift-aware benchmark (best static-under-pose-drift, same mean RMSE).
    kalman_base_r: float = 0.03          # base measurement std (m) at kalman_z_ref depth.
    kalman_z_ref: float = 1.0            # reference depth (m) for R∝(z/z_ref)^2 scaling.
    kalman_gate: float = 3.0             # innovation gate (sigmas) for outlier soft-rejection.
    yaw_smooth_window: int = 5           # circular moving-average window on per-frame box yaw
                                         # (stable 9DOF corners; 1 = off).
    # Legacy (used only when velocity_kalman=False):
    velocity_central_diff: bool = True
    smooth_window: int = 5
    use_smoothed_center: bool = True
    # Output.
    round_decimals: int = 4              # JSON numeric rounding.
    output_filename: str = "dynamic_targets.json"  # Written under the frame_dir.
    only_dynamic: bool = False           # If True, emit only objects whose track moves > min_speed.
    min_speed_dynamic: float = 0.1       # m/s; "dynamic" threshold (mean speed over track).
    # Planar-background filter (close-up segmentation cleanup). On cluttered close-up
    # scenes ~18% of tracks are large background regions (walls/floors/surfaces). Drop
    # a track only when it is BOTH big in the image AND geometrically planar — this
    # removes wall/floor regions WITHOUT dropping legit large close-up objects (which
    # have real 3D thickness, e.g. a glider filling the frame).
    # Opt-in: most large-bbox close-up tracks are REAL objects (desk/rug/glider) with
    # 3D thickness, not background — the eval showed true planar walls/floors are rare.
    # Default OFF so we never silently drop a legit thin large object (poster/screen).
    drop_planar_background: bool = False
    bg_bbox_frac: float = 0.7            # track median 2D bbox area fraction above this = "big"
    bg_min_thickness: float = 0.03       # AND thinnest 3D box axis below this (m) = planar → drop


# =============================================================================
# Main Config
# =============================================================================

@dataclass
class ROSEConfig:
    """Main ROSE pipeline configuration.

    All default values match the implementation spec
    (docs/roadmap/ROSE_IMPLEMENTATION.md, Section 6).
    """
    # Vision models
    sam3: SAM3Config = field(default_factory=SAM3Config)
    da3: DA3Config = field(default_factory=DA3Config)
    fastsam: FastSAMConfig = field(default_factory=FastSAMConfig)
    ram_plus: RAMPlusConfig = field(default_factory=RAMPlusConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Pipeline steps
    depth_filter: DepthFilterConfig = field(default_factory=DepthFilterConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    step: STEPConfig = field(default_factory=STEPConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    dynamic_targets: DynamicTargetsConfig = field(default_factory=DynamicTargetsConfig)

    # Global settings
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


# =============================================================================
# YAML Loading / Saving
# =============================================================================

def _build_from_dict(cls, raw: Dict[str, Any]):
    """Recursively construct dataclass from a dict."""
    if not isinstance(raw, dict):
        return cls()
    kwargs = {}
    for name, field_info in cls.__dataclass_fields__.items():
        if name not in raw:
            continue
        val = raw[name]
        ft = field_info.type
        # Resolve string annotations
        if isinstance(ft, str):
            import sys
            module = sys.modules.get(cls.__module__)
            ft = getattr(module, ft, ft) if module else ft
        if hasattr(ft, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[name] = _build_from_dict(ft, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_rose_config(path: Union[str, Path]) -> ROSEConfig:
    """Load ROSEConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _build_from_dict(ROSEConfig, data)


def save_rose_config(config: ROSEConfig, path: Union[str, Path]) -> None:
    """Save ROSEConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
