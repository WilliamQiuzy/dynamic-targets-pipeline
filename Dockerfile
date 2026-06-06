# ROSE — Dockerfile
# Base: RunPod PyTorch (torch 2.4 + CUDA 12.4 + cuDNN, ubuntu22.04).
# H100/H200 supported (CUDA 12.x compute capability 9.0).
#
# Build:
#   docker build -t rose .
#
# Run (interactive shell, GPU passthrough, mount local weights cache):
#   docker run --gpus all -it --rm \
#     -v $(pwd)/rose/models:/workspace/rose/rose/models \
#     rose
#
# First-time weight download inside container (requires HF login for
# gated models — Gemma, SAM3):
#   huggingface-cli login
#   bash scripts/setup.sh --core

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ── System packages ─────────────────────────────────────────────────────
# libheif-dev: required by pillow_heif when no manylinux wheel is available.
# libegl1, libgles2: open3d's optional headless render path links against them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs openssh-server rsync tmux htop wget curl \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libegl1 libgles2 libheif-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ── SSH (RunPod / remote dev convenience, no-op if unused) ──────────────
RUN mkdir -p /var/run/sshd \
    && echo "PermitRootLogin yes"        >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config

WORKDIR /workspace/rose

# ── Python deps (own layer, cached unless requirements.txt changes) ─────
COPY requirements.txt .
# blinker 1.4 in the base image is a distutils install pip refuses to uninstall;
# reinstall it via pip first so transitive upgrades from requirements.txt succeed.
RUN pip install --upgrade pip \
    && pip install --ignore-installed blinker \
    && pip install -r requirements.txt \
    && pip install hf_transfer

# ── Project code (includes vendored sam3 + da3 with ROSE patches) ───────
# 2026-05-13: rose/vision/sam3/ and rose/vision/da3/ are NOW committed to
# ROSE (with our FA3 fixes in sam3/model/model_misc.py among others).
# .dockerignore was updated to allow them in.  Weight files stay out via
# .gitignore + .dockerignore patterns (*.pt, *.pth, ..., rose/models/).
COPY . .

# ── Editable installs of vendored vision deps ───────────────────────────
# --no-deps because their pyproject.toml lists pull in opencv-python (we
# use the headless variant from requirements.txt) and other duplicates
# that would silently upgrade torch and break CUDA driver compatibility.
# Everything sam3 / da3 import at runtime is already in requirements.txt.
RUN if [ -f rose/vision/da3/pyproject.toml ] || [ -f rose/vision/da3/setup.py ]; then \
        pip install -e rose/vision/da3 --no-deps ; \
    fi \
 && if [ -f rose/vision/sam3/pyproject.toml ] || [ -f rose/vision/sam3/setup.py ]; then \
        pip install -e rose/vision/sam3 --no-deps ; \
    fi

# ── Flash Attention 3 for Hopper (H100/H200) ────────────────────────────
# sam3/perflib/fa3.py uses `flash_attn_interface` for the FP8 attention
# kernel.  Built from the `hopper/` subdir of Dao-AILab/flash-attention;
# the wheel is NOT on PyPI.  Compiles only against sm_90 (H100/H200).
#
# This add ~20-30 min to the image build, but bakes the speedup in so
# users don't have to chase a separate post-install step.  If you're
# building for non-Hopper hardware, comment out this block — the pipeline
# falls back to SDPA-flash (FA2 in bf16) without it.
#
# MAX_JOBS=4: nvcc parallelism cap.  Higher uses more RAM (each job is
# ~3-4GB during template instantiation).  4 is a safe default; bump to
# 8 on a beefy build host to halve build time.
# TORCH_CUDA_ARCH_LIST=9.0: H100/H200 only (cuts unused arch flavours).
# SKIP_FA3=1 skips this (heavy nvcc) compile — use it when building linux/amd64
# on a non-amd64 host (e.g. Apple-Silicon Mac via QEMU), where the FA3 compile is
# impractically slow/fragile under emulation. The pipeline then falls back to
# SDPA-flash (FA2 bf16) at runtime — fully functional, attention just a bit slower.
# Build FA3 in natively (recommended) on an amd64 host:  docker build .
# Skip it (fast, portable):  docker build --build-arg SKIP_FA3=1 .
ARG SKIP_FA3=0
ENV TORCH_CUDA_ARCH_LIST=9.0 \
    FLASH_ATTN_CUDA_ARCHS=90 \
    MAX_JOBS=4
RUN if [ "$SKIP_FA3" = "1" ]; then \
        echo ">>> SKIP_FA3=1 — skipping Flash-Attention-3 build (runtime falls back to FA2/SDPA)"; \
    else \
        git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
     && cd /tmp/flash-attention/hopper \
     && pip install -v --no-build-isolation . \
     && cd / \
     && rm -rf /tmp/flash-attention \
     && python -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"; \
    fi

# ── Build-time sanity: deps + vendored sam3/da3 import cleanly ──────────
# Validates that:
#   (a) torch + CUDA build is intact (no silent driver/torch upgrade),
#   (b) sam3 imports from the vendored rose/vision/sam3 (with ROSE patches),
#   (c) depth_anything_3 editable install resolves,
#   (d) rose's own package + key config dataclasses import,
#   (e) the FA3 patch in sam3/model/model_misc.py is present (qkv-same-embed
#       branch threads use_fa3) — fails fast if Dockerfile drift overwrote it.
RUN python -c "import torch, sam3, depth_anything_3; \
import rose; \
from rose.engine.config.rose_config import SAM3Config, DA3Config, FastSAMConfig, DynamicTargetsConfig; \
from rose.engine.pipeline.rose_pipeline import ROSEPipeline, FastFrameInput, FastLocalDetection; \
from rose.engine.export.obb import gravity_aligned_obb, corners_from_center_size_yaw; \
from rose.reasoning.vlm.hf_namer import HFInstanceNamer; \
from sam3.model.model_misc import MultiheadAttention; \
import inspect as _i; \
_src = _i.getsource(MultiheadAttention.forward); \
assert 'use_fa3=self.use_fa3' in _src, \
    'FA3 patch missing — Dockerfile may have re-cloned upstream sam3 over the vendored version'; \
print('torch:', torch.__version__, 'cuda build:', torch.version.cuda); \
print('sam3 from:', sam3.__file__); \
print('depth_anything_3 from:', depth_anything_3.__file__); \
print('FA3 patch present in MultiheadAttention.forward'); \
print('ROSE import OK')"

# ── Entrypoint ──────────────────────────────────────────────────────────
RUN chmod +x scripts/entrypoint.sh scripts/setup.sh
EXPOSE 22 8888
ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["bash"]
