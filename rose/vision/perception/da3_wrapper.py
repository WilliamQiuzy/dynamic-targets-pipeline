"""DA3 (Depth Anything V3) wrapper for ROSE pipeline.

Implements Step 2: monocular depth + pose estimation.

Single-frame mode (``infer``):  depth only -- T_wc ~ identity (no
inter-frame motion).

Batch mode (``infer_batch``):  all frames processed together so DA3
can estimate inter-frame camera poses.  Frame 0 is normalised to
identity; subsequent T_wc encode motion relative to frame 0.  This is
the recommended mode for building a 4DSG with consistent world
coordinates.

Chunked batch mode (``infer_batch_chunked``):  when frames exceed
``chunk_size``, splits into overlapping chunks and aligns them via
SIM3 point-cloud matching on overlap frames.  Produces output
identical in format to ``infer_batch``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from rose.engine.config.rose_config import DA3Config

logger = logging.getLogger(__name__)


def _patch_da3_fast_preprocessor(da3_model) -> None:
    """Monkey-patch DA3's ``input_processor`` so that when called with a list
    of UNIFORM numpy frames (same H,W,3, all uint8), it fast-paths the resize
    + normalize on the GPU instead of PIL CPU workers.

    Saves ~2-3s per video for typical 32-frame uniform inputs.

    The patched __call__ falls back to the original implementation when:
      - inputs aren't uniform numpy
      - extrinsics or intrinsics provided
      - process_res_method isn't ``upper_bound_resize``
    """
    proc = da3_model.input_processor
    if getattr(proc, "_fast_path_patched", False):
        return
    _orig_call = proc.__call__
    PATCH = proc.PATCH_SIZE

    import torch
    import torch.nn.functional as F

    def _fast_call(self, image, extrinsics=None, intrinsics=None,
                   process_res=504, process_res_method="upper_bound_resize",
                   *, num_workers=8, print_progress=False, sequential=None,
                   desc="Preprocess"):
        # Fast-path eligibility
        ok = (
            isinstance(image, list) and len(image) > 0
            and all(isinstance(im, np.ndarray) for im in image)
            and len({im.shape for im in image}) == 1
            and image[0].ndim == 3 and image[0].shape[2] == 3
            and image[0].dtype == np.uint8
            and extrinsics is None and intrinsics is None
            and process_res_method == "upper_bound_resize"
        )
        if not ok:
            return _orig_call(self, image, extrinsics, intrinsics, process_res, process_res_method,
                              num_workers=num_workers, print_progress=print_progress,
                              sequential=sequential, desc=desc)

        H, W = image[0].shape[:2]
        scale = float(process_res) / max(H, W)
        target_h = max(PATCH, int(round(H * scale)) // PATCH * PATCH)
        target_w = max(PATCH, int(round(W * scale)) // PATCH * PATCH)
        stacked = np.stack(image)  # (N, H, W, 3) uint8
        t = torch.from_numpy(stacked).cuda(non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous().float() / 255.0
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        t = (t - mean) / std
        # Original input_processor returns (N, 3, H, W) — _prepare_model_inputs
        # adds the batch dim via [None]. Don't unsqueeze here.
        return t, None, None  # (N, 3, H, W), exts, ixts

    # Patch at the CLASS level — instance-level __call__ bypass doesn't work
    # in Python because dunder method lookup uses the type, not instance dict.
    proc.__class__.__call__ = _fast_call
    proc._fast_path_patched = True
    logger.info("DA3 input_processor: GPU fast-path enabled (class-level)")


@dataclass
class DA3Result:
    """Output of DA3 inference for a single frame.

    Attributes:
        depth: (H, W) depth map.  Metric (metres) when the model is a nested
            metric variant; relative (arbitrary scale) otherwise.
        K: (3, 3) camera intrinsics matrix.
        T_wc: (4, 4) world-to-camera transform (p_cam = T_wc @ p_world).
        depth_conf: (H, W) per-pixel depth confidence in [0, 1].
        is_metric: Whether depth values are in absolute metres.
    """
    depth: np.ndarray
    K: np.ndarray
    T_wc: np.ndarray
    depth_conf: np.ndarray
    is_metric: bool = True

    @property
    def T_cw(self) -> np.ndarray:
        """Camera-to-world transform: p_world = T_cw @ p_cam."""
        return np.linalg.inv(self.T_wc)


class DA3Wrapper:
    """Wrapper around Depth Anything V3 for per-frame depth + pose estimation."""

    @staticmethod
    def _gpu_preprocess_uniform(np_frames, process_res: int = 504, patch: int = 14):
        """Fast path: stack numpy frames → GPU resize + normalize, all in one shot.

        Assumes all frames have the same (H, W).  Returns (1, N, 3, H', W') tensor
        where (H', W') = upper_bound_resize(orig, process_res) made divisible by patch.
        """
        import torch.nn.functional as F
        H, W = np_frames[0].shape[:2]
        scale = float(process_res) / max(H, W)
        target_h = max(patch, int(round(H * scale)) // patch * patch)
        target_w = max(patch, int(round(W * scale)) // patch * patch)
        stacked = np.stack(np_frames)  # (N, H, W, 3) uint8
        t = torch.from_numpy(stacked).cuda(non_blocking=True)
        t = t.permute(0, 3, 1, 2).contiguous().float() / 255.0  # (N, 3, H, W)
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
        t = (t - mean) / std
        return t.unsqueeze(0)  # (1, N, 3, H', W')

    def __init__(self, config: Optional[DA3Config] = None):
        self.config = config or DA3Config()
        self._model = None

    def load(self) -> None:
        """Load the DA3 model. Called lazily on first inference."""
        if self._model is not None:
            return

        import sys
        da3_src = str(Path("rose/vision/da3/src").resolve())
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)

        from depth_anything_3.api import DepthAnything3

        self._model = DepthAnything3.from_pretrained(self.config.model_path)
        self._model = self._model.to(self.config.device)
        self._model.eval()
        logger.info("DA3 model loaded from %s on %s", self.config.model_path, self.config.device)
        # GPU fast preprocessor (~13× speedup over PIL CPU path):
        # bypasses DA3's input_processor when inputs are uniform numpy/PIL frames.
        _patch_da3_fast_preprocessor(self._model)

    def unload(self) -> None:
        """Release the DA3 model from GPU memory."""
        if self._model is None:
            return
        import torch
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("DA3 model unloaded from GPU")

    def infer(self, image: Union[np.ndarray, str, Path]) -> DA3Result:
        """Run DA3 on a single image.

        Args:
            image: RGB image as (H, W, 3) uint8 ndarray, or path to image file.

        Returns:
            DA3Result with depth, intrinsics, extrinsics, and confidence
            all at the **original** image resolution.
        """
        self.load()

        import torch
        from PIL import Image as PILImage

        # Remember original resolution for later resize
        if isinstance(image, (str, Path)):
            pil_img = PILImage.open(str(image)).convert("RGB")
            orig_h, orig_w = pil_img.height, pil_img.width
            image_input = [pil_img]
        elif isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
            pil_img = PILImage.fromarray(image)
            image_input = [pil_img]
        else:
            image_input = [image]
            orig_h = orig_w = None  # cannot resize without known dims

        with torch.inference_mode():
            prediction = self._model.inference(
                image=image_input,
                process_res=self.config.process_res,
                process_res_method=self.config.process_res_method,
            )

        # --- Validate required outputs (spec Step 2) ---
        if prediction.intrinsics is None:
            raise RuntimeError(
                "DA3 did not output intrinsics (K). "
                "Ensure model_path points to a model that supports camera estimation."
            )
        if prediction.extrinsics is None:
            raise RuntimeError(
                "DA3 did not output extrinsics (T_wc). "
                "Ensure model_path points to a model that supports pose estimation."
            )
        is_metric = bool(prediction.is_metric)
        if self.config.require_metric and not is_metric:
            raise RuntimeError(
                f"DA3 model at {self.config.model_path} outputs relative depth "
                f"(is_metric=0) but require_metric=True. "
                f"Use a metric model (e.g. da3nested-giant-large) or set require_metric=False."
            )
        if not is_metric:
            logger.info("DA3 outputs relative depth (not metric). 3D scale is arbitrary.")

        # Extract single-frame results (batch dim = 0)
        depth = prediction.depth[0]  # (proc_H, proc_W)
        K = prediction.intrinsics[0]
        T_wc = prediction.extrinsics[0]

        # Guarantee T_wc is (4, 4); DA3 may return (3, 4)
        if T_wc.ndim == 2 and T_wc.shape == (3, 4):
            T_wc_44 = np.eye(4, dtype=T_wc.dtype)
            T_wc_44[:3, :] = T_wc
            T_wc = T_wc_44
        elif T_wc.shape != (4, 4):
            raise RuntimeError(
                f"DA3 extrinsics has unexpected shape {T_wc.shape}; expected (4,4) or (3,4)."
            )
        if prediction.conf is not None:
            conf = prediction.conf[0].astype(np.float32)
        else:
            conf = np.ones_like(depth, dtype=np.float32)

        # --- Resize depth/conf to original resolution (align with SAM masks) ---
        #
        # DA3 forward pipeline:
        #   orig (orig_w, orig_h)
        #     -> boundary resize -> (resized_w, resized_h)    [K scaled]
        #     -> (*crop only) center-crop to PATCH_SIZE multiples -> (proc_w, proc_h)  [K cx/cy shifted]
        #
        # We invert in reverse order: undo crop, then undo resize.
        proc_h, proc_w = depth.shape
        if orig_h is not None and (proc_h != orig_h or proc_w != orig_w):
            import cv2

            method = self.config.process_res_method
            if "crop" in method:
                # 1) Recover the intermediate resize dimensions
                process_res = self.config.process_res
                if method.startswith("upper_bound"):
                    scale = process_res / max(orig_w, orig_h)
                else:  # lower_bound
                    scale = process_res / min(orig_w, orig_h)
                resized_w = max(1, round(orig_w * scale))
                resized_h = max(1, round(orig_h * scale))

                # 2) Undo center-crop (PATCH_SIZE alignment)
                crop_x = (resized_w - proc_w) // 2
                crop_y = (resized_h - proc_h) // 2
                K = K.copy()
                K[0, 2] += crop_x  # cx
                K[1, 2] += crop_y  # cy

                # depth/conf: pad back to (resized_h, resized_w)
                pad_bottom = resized_h - proc_h - crop_y
                pad_right = resized_w - proc_w - crop_x
                depth = cv2.copyMakeBorder(
                    depth, crop_y, pad_bottom, crop_x, pad_right,
                    cv2.BORDER_REPLICATE,
                )
                conf = cv2.copyMakeBorder(
                    conf, crop_y, pad_bottom, crop_x, pad_right,
                    cv2.BORDER_CONSTANT, value=0,
                )

                # 3) Undo boundary resize: scale K and depth/conf
                #    from (resized_h, resized_w) -> (orig_h, orig_w)
                sx, sy = orig_w / resized_w, orig_h / resized_h
                K[0, :] *= sx
                K[1, :] *= sy
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                # Resize methods: scale depth/conf and adjust K proportionally
                depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                sx, sy = orig_w / proc_w, orig_h / proc_h
                K = K.copy()
                K[0, :] *= sx  # fx, s, cx
                K[1, :] *= sy  # fy, cy

        return DA3Result(
            depth=depth.astype(np.float32),
            K=K.astype(np.float64),
            T_wc=T_wc.astype(np.float64),
            depth_conf=conf,
            is_metric=is_metric,
        )

    # ------------------------------------------------------------------
    # Batch inference (consistent inter-frame poses)
    # ------------------------------------------------------------------

    def infer_batch(self, images: Sequence[Union[np.ndarray, str, Path]]) -> List[DA3Result]:
        """Run DA3 on multiple frames with consistent inter-frame poses.

        Chunk routing (``config.chunk_size``):
          - ``0``  (default): **auto** — estimate safe chunk size from free
            GPU memory so large videos never OOM.
          - ``> 0``: explicit chunk size; splits when ``len(images) > chunk_size``.
          - ``< 0``: disable chunking entirely (always single forward pass).

        Frame 0 is normalised to T_wc = identity; subsequent T_wc encode
        motion relative to frame 0.

        Args:
            images: List of RGB images (H, W, 3) uint8 ndarrays or paths.

        Returns:
            List of DA3Result, one per frame, with consistent depth, K,
            T_wc across all frames.
        """
        if len(images) == 0:
            return []
        if len(images) == 1:
            return [self.infer(images[0])]

        cs = self.config.chunk_size
        if cs == 0:
            # Auto-detect safe chunk size based on available GPU memory
            cs = self._estimate_safe_chunk_size(len(images))
        elif cs < 0:
            # Negative = disable chunking entirely
            cs = len(images)

        if len(images) > cs:
            return self.infer_batch_chunked(images, chunk_size=cs)

        return self._infer_batch_core(images)

    def _estimate_safe_chunk_size(self, n_frames: int) -> int:
        """Estimate max frames per DA3 batch based on available GPU memory.

        Memory model (empirical, from DA3_BATCH_OOM.md):
          base ≈ 2 GB  (model weights + CUDA context)
          per_frame ≈ (process_res / 504)² × 0.5 GB

        Targets 50% of free GPU memory (conservative to avoid OOM retry).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return n_frames

            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
        except Exception:
            logger.warning("Cannot query GPU memory; defaulting chunk_size=16")
            return min(16, n_frames)

        scale = (self.config.process_res / 504) ** 2
        per_frame_gb = scale * 0.5
        base_gb = 2.0
        target_usage = 0.50

        available_gb = free_gb * target_usage - base_gb
        if available_gb <= per_frame_gb:
            safe_n = 4  # minimum viable batch
        else:
            safe_n = int(available_gb / per_frame_gb)

        min_chunk = self.config.chunk_overlap + 1
        safe_n = max(min_chunk, min(safe_n, n_frames))

        logger.info(
            "DA3 auto chunk_size=%d (free=%.1fGB, per_frame=%.2fGB, n_frames=%d)",
            safe_n, free_gb, per_frame_gb, n_frames,
        )
        return safe_n

    def _infer_batch_core(
        self, images: Sequence[Union[np.ndarray, str, Path]],
    ) -> List[DA3Result]:
        """Run DA3 on multiple frames in a single forward pass (original path).

        This is the internal implementation that always sends all frames to
        DA3 in one batch.  For the public API with auto-chunking, use
        ``infer_batch``.

        If the forward pass triggers ``torch.cuda.OutOfMemoryError``, it
        automatically falls back to chunked inference with half the batch.
        """
        self.load()

        import torch
        from PIL import Image as PILImage

        # Reclaim stale GPU memory before the heavy allocation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Detect uniform numpy fast-path eligibility — pass raw numpy through
        # so the patched input_processor's GPU fast-path can fire.  Otherwise
        # convert to PIL as before.
        np_uniform = (
            isinstance(images, list) and len(images) > 0
            and all(isinstance(im, np.ndarray) for im in images)
            and len({im.shape for im in images}) == 1
            and images[0].ndim == 3 and images[0].shape[2] == 3
            and images[0].dtype == np.uint8
        )
        orig_sizes: List[Tuple[int, int]] = []
        if np_uniform:
            pil_images = list(images)  # numpy passthrough; processor will fast-path
            for img in images:
                orig_sizes.append((img.shape[0], img.shape[1]))
        else:
            pil_images: List = []
            for img in images:
                if isinstance(img, (str, Path)):
                    pil = PILImage.open(str(img)).convert("RGB")
                elif isinstance(img, np.ndarray):
                    pil = PILImage.fromarray(img)
                else:
                    pil = img
                orig_sizes.append((pil.height, pil.width))
                pil_images.append(pil)

        try:
            with torch.inference_mode():
                prediction = self._model.inference(
                    image=pil_images,
                    process_res=self.config.process_res,
                    process_res_method=self.config.process_res_method,
                )
        except torch.cuda.OutOfMemoryError:
            # OOM safety net: fall back to chunked inference with half batch
            torch.cuda.empty_cache()
            n = len(pil_images)
            min_chunk = self.config.chunk_overlap + 1
            if n <= min_chunk:
                raise  # Already too small to chunk
            half = max(min_chunk, n // 2)
            logger.warning(
                "DA3 OOM with %d frames in single batch; "
                "falling back to chunked inference (chunk_size=%d)",
                n, half,
            )
            return self.infer_batch_chunked(images, chunk_size=half)

        # Validate
        if prediction.intrinsics is None:
            raise RuntimeError("DA3 did not output intrinsics (K).")
        if prediction.extrinsics is None:
            raise RuntimeError("DA3 did not output extrinsics (T_wc).")
        is_metric = bool(prediction.is_metric)
        if self.config.require_metric and not is_metric:
            raise RuntimeError(
                f"DA3 model outputs relative depth but require_metric=True."
            )

        n_frames = len(pil_images)

        # Extract and guarantee (4, 4) for all extrinsics first
        all_T_wc: List[np.ndarray] = []
        for i in range(n_frames):
            T_wc = prediction.extrinsics[i]
            if T_wc.ndim == 2 and T_wc.shape == (3, 4):
                T_wc_44 = np.eye(4, dtype=T_wc.dtype)
                T_wc_44[:3, :] = T_wc
                T_wc = T_wc_44
            elif T_wc.shape != (4, 4):
                raise RuntimeError(
                    f"DA3 extrinsics[{i}] has unexpected shape {T_wc.shape}."
                )
            all_T_wc.append(T_wc.astype(np.float64))

        # Normalize: make frame 0 exactly identity (world = camera-0 frame).
        # T_wc_new[i] = T_wc_raw[i] @ inv(T_wc_raw[0])
        T0_inv = np.linalg.inv(all_T_wc[0])
        for i in range(n_frames):
            all_T_wc[i] = all_T_wc[i] @ T0_inv

        results: List[DA3Result] = []

        for i in range(n_frames):
            depth = prediction.depth[i]
            K = prediction.intrinsics[i]
            T_wc = all_T_wc[i]

            conf = (
                prediction.conf[i].astype(np.float32)
                if prediction.conf is not None
                else np.ones_like(depth, dtype=np.float32)
            )

            # Resize depth/conf to this frame's original resolution
            orig_h, orig_w = orig_sizes[i]
            proc_h, proc_w = depth.shape
            if proc_h != orig_h or proc_w != orig_w:
                depth, conf, K = self._resize_to_original(
                    depth, conf, K, orig_h, orig_w, proc_h, proc_w,
                )

            results.append(DA3Result(
                depth=depth.astype(np.float32),
                K=K.astype(np.float64),
                T_wc=T_wc.astype(np.float64),
                depth_conf=conf,
                is_metric=is_metric,
            ))

        return results

    # ------------------------------------------------------------------
    # Chunked batch inference (SIM3-aligned overlapping chunks)
    # ------------------------------------------------------------------

    def infer_batch_chunked(
        self,
        images: Sequence[Union[np.ndarray, str, Path]],
        chunk_size: Optional[int] = None,
    ) -> List[DA3Result]:
        """Run DA3 in overlapping chunks with SIM3 pose alignment.

        Splits frames into overlapping chunks, runs DA3 batch inference
        on each chunk independently, then aligns all chunks to chunk 0's
        coordinate system via SIM3 point-cloud matching on overlap frames.

        The output format is identical to ``infer_batch``: frame 0 is
        normalised to T_wc = identity.

        Args:
            images: Sequence of RGB images or paths.
            chunk_size: Override for ``config.chunk_size``.  When provided,
                this value is used instead of the config default.  This is
                used by the auto-detection and OOM-fallback paths.

        Returns:
            List of DA3Result, one per frame, with chunk-aligned poses.
        """
        import torch

        images_list = list(images)
        n = len(images_list)
        chunk_size = chunk_size if chunk_size is not None else self.config.chunk_size
        overlap = self.config.chunk_overlap

        if overlap < 2:
            raise ValueError(
                f"chunk_overlap must be >= 2 for SIM3 alignment, got {overlap}"
            )
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )

        chunks = compute_chunks(n, chunk_size, overlap)

        if len(chunks) == 1:
            # Single chunk — still pad to chunk_size so the tensor shape
            # matches the warmup batch (avoids CUDA JIT recompilation).
            pad_count = chunk_size - n
            if pad_count > 0:
                padded = list(images_list) + [images_list[-1]] * pad_count
                results = self._infer_batch_core(padded)
                return results[:n]
            return self._infer_batch_core(images_list)

        logger.info(
            "Chunked DA3: %d frames -> %d chunks (size=%d, overlap=%d)",
            n, len(chunks), chunk_size, overlap,
        )

        step = chunk_size - overlap

        # ------ Phase 1: run DA3 on each chunk ------
        # Pad short last chunk to chunk_size by repeating the final frame.
        # This ensures all forward passes use the same batch size, avoiding
        # costly CUDA kernel JIT recompilation for a different tensor shape.
        chunk_results: List[List[DA3Result]] = []
        for ci, (start, end) in enumerate(chunks):
            chunk_frames = images_list[start:end]
            actual_len = len(chunk_frames)
            pad_count = chunk_size - actual_len
            if pad_count > 0:
                chunk_frames = list(chunk_frames) + [chunk_frames[-1]] * pad_count
            logger.info(
                "  Chunk %d/%d: frames [%d, %d) (%d frames%s)",
                ci + 1, len(chunks), start, end, actual_len,
                f", padded to {chunk_size}" if pad_count > 0 else "",
            )
            results = self._infer_batch_core(chunk_frames)
            # Strip padding results
            if pad_count > 0:
                results = results[:actual_len]
            chunk_results.append(results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------ Phase 2: pairwise SIM3 between adjacent chunks ------
        pairwise_sim3: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for ci in range(len(chunks) - 1):
            prev_start, prev_end = chunks[ci]
            curr_start, curr_end = chunks[ci + 1]
            actual_overlap = prev_end - curr_start

            s, R, t = _align_overlap(
                chunk_results[ci][-actual_overlap:],
                chunk_results[ci + 1][:actual_overlap],
            )
            pairwise_sim3.append((s, R, t))
            logger.info(
                "  SIM3 chunk %d->%d: s=%.4f", ci + 1, ci, s,
            )

        # ------ Phase 3: accumulate to chunk 0 frame ------
        cumulative = _accumulate_sim3(pairwise_sim3)

        # ------ Phase 4: assign frames with overlap blending ------
        # 4a) Apply SIM3 to all chunk results.
        aligned_chunks: List[List[DA3Result]] = []
        for ci, (start, end) in enumerate(chunks):
            s_cum, R_cum, t_cum = cumulative[ci]
            aligned: List[DA3Result] = []
            for r in chunk_results[ci]:
                aligned.append(
                    _apply_sim3_to_result(r, s_cum, R_cum, t_cum) if ci > 0 else r
                )
            aligned_chunks.append(aligned)

        # 4b) For each frame, find which chunks contain it and blend if overlap.
        final: List[Optional[DA3Result]] = [None] * n
        for f in range(n):
            sources: List[Tuple[int, int]] = []  # (chunk_idx, local_idx)
            for ci, (start, end) in enumerate(chunks):
                if start <= f < end:
                    sources.append((ci, f - start))

            if len(sources) == 1:
                ci, li = sources[0]
                final[f] = aligned_chunks[ci][li]
            elif len(sources) >= 2:
                # Blend between the two adjacent chunks.
                ci_a, li_a = sources[0]
                ci_b, li_b = sources[1]
                overlap_start = chunks[ci_b][0]
                overlap_end = chunks[ci_a][1]
                overlap_len = overlap_end - overlap_start
                # alpha in (0, 1): smooth transition from chunk A to chunk B.
                alpha = (f - overlap_start + 1) / (overlap_len + 1)
                final[f] = _blend_da3_results(
                    aligned_chunks[ci_a][li_a],
                    aligned_chunks[ci_b][li_b],
                    alpha,
                )

        # Verify completeness
        for i, r in enumerate(final):
            if r is None:
                raise RuntimeError(
                    f"Frame {i} was not assigned to any chunk "
                    f"(chunks={chunks}, step={step})"
                )

        # ------ Phase 5: normalise frame 0 to identity ------
        # Chunk 0 already normalised frame 0 to identity in _infer_batch_core,
        # and we did not apply SIM3 to chunk 0, so this should be a no-op.
        # We do it explicitly for safety.
        T0_inv = np.linalg.inv(final[0].T_wc)
        if not np.allclose(T0_inv, np.eye(4), atol=1e-6):
            for i in range(n):
                r = final[i]
                final[i] = DA3Result(
                    depth=r.depth,
                    K=r.K,
                    T_wc=(r.T_wc @ T0_inv).astype(np.float64),
                    depth_conf=r.depth_conf,
                    is_metric=r.is_metric,
                )

        return final  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize_to_original(
        self,
        depth: np.ndarray,
        conf: np.ndarray,
        K: np.ndarray,
        orig_h: int,
        orig_w: int,
        proc_h: int,
        proc_w: int,
    ):
        """Resize depth/conf from DA3 processing resolution to original and adjust K."""
        import cv2

        method = self.config.process_res_method
        K = K.copy()

        if "crop" in method:
            process_res = self.config.process_res
            if method.startswith("upper_bound"):
                scale = process_res / max(orig_w, orig_h)
            else:
                scale = process_res / min(orig_w, orig_h)
            resized_w = max(1, round(orig_w * scale))
            resized_h = max(1, round(orig_h * scale))

            crop_x = (resized_w - proc_w) // 2
            crop_y = (resized_h - proc_h) // 2
            K[0, 2] += crop_x
            K[1, 2] += crop_y

            pad_bottom = resized_h - proc_h - crop_y
            pad_right = resized_w - proc_w - crop_x
            depth = cv2.copyMakeBorder(
                depth, crop_y, pad_bottom, crop_x, pad_right,
                cv2.BORDER_REPLICATE,
            )
            conf = cv2.copyMakeBorder(
                conf, crop_y, pad_bottom, crop_x, pad_right,
                cv2.BORDER_CONSTANT, value=0,
            )

            sx, sy = orig_w / resized_w, orig_h / resized_h
            K[0, :] *= sx
            K[1, :] *= sy
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            conf = cv2.resize(conf, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            sx, sy = orig_w / proc_w, orig_h / proc_h
            K[0, :] *= sx
            K[1, :] *= sy

        return depth, conf, K


# ======================================================================
# Module-level helpers (pure functions, importable for testing)
# ======================================================================

def compute_chunks(
    n: int, chunk_size: int, overlap: int,
) -> List[Tuple[int, int]]:
    """Compute (start, end) chunk boundaries for *n* frames.

    Adjacent chunks overlap by *overlap* frames.  If the last chunk would
    have fewer than *overlap* frames it is merged into the previous one.
    """
    step = chunk_size - overlap
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end))
        if end >= n:
            break
        start += step

    # Merge a tiny trailing chunk into the previous one.
    if len(chunks) >= 2:
        last_len = chunks[-1][1] - chunks[-1][0]
        if last_len <= overlap:
            merged_end = chunks[-1][1]
            chunks.pop()
            chunks[-1] = (chunks[-1][0], merged_end)

    return chunks


def backproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
) -> np.ndarray:
    """Backproject a depth map to 3D world coordinates.

    Args:
        depth: (H, W) depth map (float).
        K: (3, 3) camera intrinsics.
        T_wc: (4, 4) world-to-camera transform.

    Returns:
        (H, W, 3) world-coordinate point cloud.
    """
    H, W = depth.shape
    K_inv = np.linalg.inv(K[:3, :3].astype(np.float64))
    T_cw = np.linalg.inv(T_wc.astype(np.float64))
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)

    rays = (K_inv @ pixels.reshape(-1, 3).T).T.reshape(H, W, 3)
    points_cam = rays * depth[..., None].astype(np.float64)
    points_world = (R_cw @ points_cam.reshape(-1, 3).T).T + t_cw
    return points_world.reshape(H, W, 3).astype(np.float64)


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate SIM3 via Umeyama: target ~ s * R @ source + t.

    Args:
        source: (N, 3) points to be transformed.
        target: (N, 3) reference points.
        weights: (N,) optional per-point weights >= 0.

    Returns:
        (s, R, t) where s is scale, R is (3,3) rotation, t is (3,) translation.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = len(source)
    if n < 3:
        return 1.0, np.eye(3), np.zeros(3)

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    w_sum = w.sum()
    if w_sum < 1e-12:
        return 1.0, np.eye(3), np.zeros(3)

    mu_s = (w[:, None] * source).sum(axis=0) / w_sum
    mu_t = (w[:, None] * target).sum(axis=0) / w_sum

    src_c = source - mu_s
    tgt_c = target - mu_t

    # Weighted covariance
    Sigma = (tgt_c * w[:, None]).T @ src_c / w_sum
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    var_src = (w[:, None] * src_c ** 2).sum() / w_sum
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0

    t = mu_t - s * R @ mu_s
    return s, R, t


def estimate_sim3_robust(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
    delta: float = 0.5,
    max_iters: int = 5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Robust SIM3 via IRLS with Huber loss.

    First runs Umeyama, then iteratively reweights outliers.
    """
    s, R, t = estimate_sim3(source, target, weights)

    for _ in range(max_iters):
        transformed = s * (R @ source.T).T + t
        residuals = np.linalg.norm(transformed - target, axis=1)

        huber_w = np.where(residuals < delta, 1.0, delta / (residuals + 1e-12))
        combined = huber_w if weights is None else weights * huber_w

        s_new, R_new, t_new = estimate_sim3(source, target, combined)
        if abs(s_new - s) < 1e-8 and np.allclose(R_new, R, atol=1e-8):
            break
        s, R, t = s_new, R_new, t_new

    return s, R, t


def _accumulate_sim3(
    pairwise: List[Tuple[float, np.ndarray, np.ndarray]],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Convert pairwise SIM3 [chunk1->0, chunk2->1, ...] to cumulative.

    cumulative[0] = identity (chunk 0 is the reference).
    cumulative[j] maps chunk j's world -> chunk 0's world.
    """
    cumulative: List[Tuple[float, np.ndarray, np.ndarray]] = [
        (1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
    ]
    for s, R, t in pairwise:
        s_prev, R_prev, t_prev = cumulative[-1]
        s_cum = s_prev * s
        R_cum = R_prev @ R
        t_cum = s_prev * (R_prev @ t) + t_prev
        cumulative.append((s_cum, R_cum, t_cum))
    return cumulative


def _apply_sim3_to_result(
    result: DA3Result,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> DA3Result:
    """Apply cumulative SIM3 to a DA3Result.

    Transforms the result from a source chunk's coordinate system into the
    target (chunk 0) coordinate system.

    SIM3 semantics:  p_target = s * R @ p_source + t

    We scale depth by *s* so that backprojection with the new T_wc yields
    correct world-coordinate points:
        p_cam  = K^{-1} * [u,v,1] * (s * depth)
        p_world = R_cw_new @ p_cam + t_cw_new
                = s * R_sim3 @ (R_cw @ K^{-1}*[u,v,1]*depth + t_cw) + t_sim3
    """
    T_cw = np.linalg.inv(result.T_wc.astype(np.float64))
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    R_cw_new = R @ R_cw
    t_cw_new = s * (R @ t_cw) + t

    T_cw_new = np.eye(4, dtype=np.float64)
    T_cw_new[:3, :3] = R_cw_new
    T_cw_new[:3, 3] = t_cw_new
    T_wc_new = np.linalg.inv(T_cw_new)

    return DA3Result(
        depth=(result.depth * float(s)).astype(np.float32),
        K=result.K.copy(),
        T_wc=T_wc_new.astype(np.float64),
        depth_conf=result.depth_conf.copy(),
        is_metric=result.is_metric,
    )


def _blend_da3_results(a: DA3Result, b: DA3Result, alpha: float) -> DA3Result:
    """Blend two DA3Results for smooth overlap transitions.

    Args:
        a: Result from the earlier chunk.
        b: Result from the later chunk (already SIM3-aligned to a's frame).
        alpha: Blending weight in (0, 1).  0 → all *a*, 1 → all *b*.

    Returns:
        Blended DA3Result with interpolated pose, depth, and intrinsics.
    """
    # Blend depth (both already in same scale after SIM3)
    depth = ((1 - alpha) * a.depth + alpha * b.depth).astype(np.float32)

    # Confidence: conservative (minimum)
    conf = np.minimum(a.depth_conf, b.depth_conf)

    # Intrinsics: average (should be nearly identical)
    K = ((1 - alpha) * a.K + alpha * b.K).astype(np.float64)

    # Pose: interpolate in camera-to-world space
    T_cw_a = np.linalg.inv(a.T_wc.astype(np.float64))
    T_cw_b = np.linalg.inv(b.T_wc.astype(np.float64))

    # Translation: linear interpolation
    t_blend = (1 - alpha) * T_cw_a[:3, 3] + alpha * T_cw_b[:3, 3]

    # Rotation: linear blend + SVD re-orthogonalisation
    R_blend = (1 - alpha) * T_cw_a[:3, :3] + alpha * T_cw_b[:3, :3]
    U, _, Vt = np.linalg.svd(R_blend)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R_blend = U @ S @ Vt

    T_cw_blend = np.eye(4, dtype=np.float64)
    T_cw_blend[:3, :3] = R_blend
    T_cw_blend[:3, 3] = t_blend
    T_wc_blend = np.linalg.inv(T_cw_blend)

    return DA3Result(
        depth=depth,
        K=K,
        T_wc=T_wc_blend,
        depth_conf=conf,
        is_metric=a.is_metric,
    )


def _align_overlap(
    prev_overlap: List[DA3Result],
    curr_overlap: List[DA3Result],
    conf_threshold: float = 0.2,
    depth_max: float = 100.0,
    max_points: int = 100_000,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute SIM3 from curr chunk's world -> prev chunk's world.

    Uses the overlap frames (same physical frames in both chunks) to
    build corresponding 3D point clouds and run robust SIM3 estimation.
    """
    all_prev_pts: List[np.ndarray] = []
    all_curr_pts: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []

    for prev_r, curr_r in zip(prev_overlap, curr_overlap):
        pp = backproject_depth(prev_r.depth, prev_r.K, prev_r.T_wc)
        cp = backproject_depth(curr_r.depth, curr_r.K, curr_r.T_wc)
        w = np.minimum(prev_r.depth_conf, curr_r.depth_conf).astype(np.float64)

        all_prev_pts.append(pp.reshape(-1, 3))
        all_curr_pts.append(cp.reshape(-1, 3))
        all_weights.append(w.reshape(-1))

    prev_pts = np.concatenate(all_prev_pts)
    curr_pts = np.concatenate(all_curr_pts)
    weights = np.concatenate(all_weights)

    # Filter: confidence, finite values, depth range
    valid = (
        (weights > conf_threshold)
        & np.isfinite(prev_pts).all(axis=1)
        & np.isfinite(curr_pts).all(axis=1)
        & (np.linalg.norm(prev_pts, axis=1) < depth_max)
        & (np.linalg.norm(curr_pts, axis=1) < depth_max)
    )
    prev_pts = prev_pts[valid]
    curr_pts = curr_pts[valid]
    weights = weights[valid]

    if len(prev_pts) < 10:
        logger.warning(
            "Only %d valid point pairs for SIM3 alignment -- falling back "
            "to identity (no scale/rotation/translation correction)",
            len(prev_pts),
        )
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    # Subsample for speed
    rng = np.random.default_rng(42)
    if len(prev_pts) > max_points:
        idx = rng.choice(len(prev_pts), max_points, replace=False)
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        weights = weights[idx]

    # SIM3: curr_world -> prev_world
    # Prefer GPU-accelerated alignment from DA3 streaming if available.
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            from rose.vision.da3.da3_streaming.loop_utils.alignment_torch import (
                robust_weighted_estimate_sim3_torch,
            )
            s, R, t = robust_weighted_estimate_sim3_torch(
                curr_pts.astype(np.float32),
                prev_pts.astype(np.float32),
                weights.astype(np.float32),
                delta=0.1,
                max_iters=20,
                align_method="sim3",
            )
            s = float(s)
            R = R.astype(np.float64)
            t = t.astype(np.float64)
            logger.debug("SIM3 alignment used torch GPU backend")
        else:
            raise ImportError("no CUDA")
    except (ImportError, Exception) as exc:
        logger.debug("SIM3 torch backend unavailable (%s), using numpy", exc)
        s, R, t = estimate_sim3_robust(curr_pts, prev_pts, weights)

    # Sanity: reject degenerate transforms
    if s < 0.01 or s > 100.0:
        logger.warning(
            "SIM3 scale %.4f is suspicious -- clamping to identity", s,
        )
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    return s, R, t
