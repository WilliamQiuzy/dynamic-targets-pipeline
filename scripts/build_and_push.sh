#!/usr/bin/env bash
# ============================================================================
# Build the dynamic-targets image and push it PUBLIC to Docker Hub.
#
# Run this on a machine with Docker (your Mac is fine — no GPU needed to build).
# It builds for linux/amd64 (the architecture of the GPU hosts that will RUN it)
# and pushes in one step via buildx.
#
# Defaults are set for this project:
#   DOCKER_USER=ziyueqiu  IMAGE=dynamic-targets  TAG=latest  SKIP_FA3=1
# Override any of them inline, e.g.  TAG=v1 bash scripts/build_and_push.sh
#
# SKIP_FA3=1 (default) skips the Flash-Attention-3 compile so the cross-arch
# build (amd64-on-Apple-Silicon, under QEMU) is fast and reliable; the image
# still runs on any CUDA GPU (attention falls back to FA2/SDPA). To bake FA3 in
# for max speed on H100/H200, run on an amd64 + Hopper host with SKIP_FA3=0.
#
# NOTE: a PUBLIC image exposes the code baked into it. Model weights are NOT
# included (downloaded at runtime via scripts/setup.sh) — gated weights
# (facebook/sam3, gemma) must never be baked into a public image.
# ============================================================================
set -euo pipefail

DOCKER_USER="${DOCKER_USER:-ziyueqiu}"
IMAGE="${IMAGE:-dynamic-targets}"
TAG="${TAG:-latest}"
SKIP_FA3="${SKIP_FA3:-1}"
PLATFORM="${PLATFORM:-linux/amd64}"
REF="${DOCKER_USER}/${IMAGE}:${TAG}"

cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root

echo "==> Logging in to Docker Hub as ${DOCKER_USER} (use a Docker Hub access token as the password)"
docker login -u "${DOCKER_USER}"

# Ensure a buildx builder exists (needed for cross-platform build on a Mac).
docker buildx inspect rosebuilder >/dev/null 2>&1 || docker buildx create --name rosebuilder --use
docker buildx use rosebuilder

echo "==> Building ${REF}  (platform=${PLATFORM}, SKIP_FA3=${SKIP_FA3}) and pushing"
docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg "SKIP_FA3=${SKIP_FA3}" \
  -t "${REF}" \
  --push .

echo ""
echo "Done. Pushed ${REF}."
echo "Make sure the repo is PUBLIC: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE}  (Settings -> Visibility -> Public)"
echo ""
echo "Anyone with a CUDA GPU can then run:"
echo "  docker pull ${REF}"
echo "  docker run --gpus all -it --rm -v \$(pwd)/models:/workspace/rose/rose/models ${REF}"
echo "  # inside: huggingface-cli login && bash scripts/setup.sh --core"
echo "  #         python scripts/run_dynamic_targets.py video.mp4 -o out.json"
