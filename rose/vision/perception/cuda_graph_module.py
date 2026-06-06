"""Manual, shape-keyed CUDA Graph capture/replay for a module forward.

Why this exists
---------------
The SAM 3.1 multiplex tracker is *latency-bound*: during propagation the GPU is
~50% idle and per-frame kernel-launch overhead dominates (memory-attention +
mask-decoder fire hundreds of tiny kernels per frame).  ``torch.compile`` with
``dynamic=True`` (our prod default) only *fuses* kernels — it DISABLES CUDA
graphs on dynamic shapes — so it captures none of the launch-overhead win
(measured ~4%).  ``torch.compile(dynamic=False)`` was tested and is SLOWER: it
re-runs the expensive Inductor compile on every new shape and thrashes.

This wrapper takes the industrial route NVIDIA documents for dynamic patterns:
capture a real CUDA graph per distinct input-shape *signature*, copy each call's
inputs into persistent static buffers, and ``replay()`` — collapsing hundreds of
launches into one.  Re-capture on a new shape is cheap (one forward), unlike
Inductor recompile.  The SAM 3.1 memory bank GROWS for the first ~7 frames then
STABILISES, so a handful of graphs (steady-state shape + a few ramp-up / bucket
counts) covers almost every frame.

It is **bit-exact** (same kernels, merely replayed) and bf16-safe (CUDA graphs
are precision-agnostic, unlike torch-tensorrt).

Contract / assumptions
----------------------
* The wrapped forward must be free of GPU->CPU syncs (``.item()``/``.cpu()``/
  data-dependent control flow).  SAM 3.1's ``decoder.py`` hot path satisfies this
  (only ``.shape[...]`` int comparisons, which are baked in at capture).
* Tensor kwargs are static-buffered; non-tensor kwargs (ints, tuples, None) are
  treated as compile-time constants and folded into the signature.
* Output may be a Tensor or a (possibly nested) dict/list/tuple of Tensors; all
  output tensors are cloned before return so the next replay can't clobber them.
* Calls are sequential (one frame fully consumed before the next) — true for
  SAM 3 propagation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _sig_of(x: Any) -> Any:
    """Signature element for one argument."""
    if _is_tensor(x):
        return ("T", tuple(x.shape), x.dtype, x.device.type)
    if isinstance(x, (list, tuple)):
        return ("seq", type(x).__name__, tuple(_sig_of(e) for e in x))
    # ints / floats / bools / None / strings -> constants
    return ("c", x)


class _Captured:
    """One captured graph + its static input/output buffers for a fixed sig."""

    __slots__ = ("graph", "static_in", "static_out", "pool")

    def __init__(self, graph, static_in, static_out, pool):
        self.graph = graph
        self.static_in = static_in        # dict name -> static tensor (tensor kwargs only)
        self.static_out = static_out      # the captured output structure (tensors are static)
        self.pool = pool


class CUDAGraphedModule:
    """Wrap ``module.forward`` with shape-keyed manual CUDA graph capture.

    Use ``module.forward = CUDAGraphedModule(module)`` (it is callable and stores
    the original forward).  Pass-through (eager) for shapes beyond ``max_graphs``
    or if capture fails — never wrong, just unaccelerated.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        warmup_iters: int = 3,
        max_graphs: int = 16,
        capture_threshold: int = 2,
        name: str = "module",
        enabled: bool = True,
    ):
        self._module = module
        self._orig_forward = module.forward
        self._warmup_iters = int(warmup_iters)
        self._max_graphs = int(max_graphs)
        # Only capture a shape once we've seen it >= capture_threshold times.
        # Ramp-up frames each have a unique memory length (seen once) -> stay
        # eager; the frequent STEADY-STATE shape repeats -> gets captured. This
        # stops one-off shapes from burning graph slots / capture time.
        self._capture_threshold = int(capture_threshold)
        self._name = name
        self._enabled = bool(enabled)
        self._graphs: Dict[Any, _Captured] = {}
        self._seen: Dict[Any, int] = {}
        self._cuda_ok = torch.cuda.is_available()
        # diagnostics
        self.n_replay = 0
        self.n_eager = 0
        self.n_capture = 0

    # ---- public stats -------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "graphs": len(self._graphs),
            "replays": self.n_replay,
            "eager": self.n_eager,
            "captures": self.n_capture,
        }

    # ---- signature ----------------------------------------------------
    def _signature(self, kwargs: Dict[str, Any]) -> Any:
        return tuple(sorted((k, _sig_of(v)) for k, v in kwargs.items()))

    # ---- main call ----------------------------------------------------
    def __call__(self, *args, **kwargs):
        # We require keyword-only calls (SAM 3.1 calls the encoder with kwargs).
        # If positional args appear, fall back to eager — never risk correctness.
        if args or not self._enabled or not self._cuda_ok:
            self.n_eager += 1
            return self._orig_forward(*args, **kwargs)

        try:
            sig = self._signature(kwargs)
        except Exception:
            self.n_eager += 1
            return self._orig_forward(**kwargs)

        entry = self._graphs.get(sig)
        if entry is None:
            # Count occurrences; only capture once a shape has REPEATED enough
            # (steady-state), so one-off ramp-up shapes don't burn graph slots.
            seen = self._seen.get(sig, 0) + 1
            self._seen[sig] = seen
            if seen < self._capture_threshold or len(self._graphs) >= self._max_graphs:
                self.n_eager += 1
                return self._orig_forward(**kwargs)
            entry = self._try_capture(kwargs, sig)
            if entry is None:
                self.n_eager += 1
                return self._orig_forward(**kwargs)
            self._graphs[sig] = entry

        # cache hit: copy dynamic tensor inputs into the static buffers, replay.
        for k, static_t in entry.static_in.items():
            static_t.copy_(kwargs[k], non_blocking=True)
        entry.graph.replay()
        self.n_replay += 1
        return _clone_structure(entry.static_out)

    # ---- capture ------------------------------------------------------
    def _try_capture(self, kwargs: Dict[str, Any], sig: Any) -> Optional[_Captured]:
        try:
            return self._capture(kwargs, sig)
        except Exception as e:  # capture is best-effort; eager is always correct
            logger.warning(
                "[cudagraph:%s] capture failed (falling back to eager for this shape): %s",
                self._name, e,
            )
            return None

    def _capture(self, kwargs: Dict[str, Any], sig: Any) -> _Captured:
        # 1) Build STATIC input buffers: a persistent clone of each tensor kwarg.
        #    Non-tensor kwargs are passed through verbatim (constants).
        static_in: Dict[str, torch.Tensor] = {}
        call_kwargs: Dict[str, Any] = {}
        for k, v in kwargs.items():
            if _is_tensor(v):
                static_in[k] = v.detach().clone()
                call_kwargs[k] = static_in[k]
            else:
                call_kwargs[k] = v

        # 2) Warm up on a SIDE stream so cuDNN/allocator pick stable workspaces
        #    BEFORE capture (required by the CUDA graph capture protocol).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(1, self._warmup_iters)):
                _ = self._orig_forward(**call_kwargs)
        torch.cuda.current_stream().wait_stream(s)

        # 3) Capture.  capture_error_mode="thread_local" is CRITICAL here: the
        #    pipeline runs DA3 depth concurrently on a SEPARATE host thread +
        #    CUDA stream.  Under the default "global" capture mode, that thread's
        #    allocator activity (cudaMalloc/Free/eventQuery) would invalidate our
        #    capture (cudaErrorStreamCaptureInvalidated) or corrupt it.
        #    thread_local restricts capture to THIS thread, letting DA3 keep
        #    running.  (NVIDIA CUDA-graph capture-failures guidance.)
        # PER-GRAPH private pool: avoids the "graphs sharing one pool must replay
        # in capture order / never concurrently" hazard (agent-1 finding). Our
        # shapes interleave across frames (ramp-up vs steady), so a shared pool
        # is unsafe; a dedicated pool per shape-key removes the ordering coupling.
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, pool=pool, capture_error_mode="thread_local",
        ):
            static_out = self._orig_forward(**call_kwargs)

        self.n_capture += 1
        logger.info(
            "[cudagraph:%s] captured graph #%d (sig has mem=%s)",
            self._name, len(self._graphs) + 1,
            _first_tensor_shapes(kwargs),
        )
        return _Captured(graph, static_in, static_out, pool)


def _clone_structure(x: Any) -> Any:
    """Deep-clone tensors inside a Tensor / dict / list / tuple structure."""
    if _is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_structure(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        cloned = [_clone_structure(v) for v in x]
        return type(x)(cloned) if not isinstance(x, tuple) else tuple(cloned)
    return x  # non-tensor passthrough (e.g. None, ints)


def _first_tensor_shapes(kwargs: Dict[str, Any]) -> str:
    parts = []
    for k, v in kwargs.items():
        if _is_tensor(v):
            parts.append(f"{k}{tuple(v.shape)}")
    return ", ".join(parts[:6])
