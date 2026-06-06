#!/usr/bin/env bash
# ============================================================================
# ROSE One-Click Setup Script
#
# Downloads all external code repositories, model weights, and benchmark data
# that are gitignored from the main repository.
#
# Usage:
#   bash scripts/setup.sh          # Install everything
#   bash scripts/setup.sh --core   # Core pipeline only (no benchmark)
#
# Prerequisites:
#   - conda environment "snow" with PyTorch, transformers, etc.
#   - git, git-lfs
#   - huggingface-cli logged in (for gated models like SAM3, Gemma)
#     Run: huggingface-cli login
# ============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# Parse arguments
# ============================================================================
INSTALL_CORE=true
INSTALL_BENCH=true

if [[ "${1:-}" == "--core" ]]; then
    INSTALL_BENCH=false
    INSTALL_CORE=false
fi

# ============================================================================
# Helper: clone repo if not already present
# ============================================================================
clone_repo() {
    local url="$1"
    local target="$2"
    local name
    name="$(basename "$target")"

    if [ -d "$target/.git" ]; then
        info "$name already cloned (has .git/), skipping"
    elif [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        # 2026-05-13: sam3 and da3 are now vendored directly inside ROSE
        # (with patches applied — see commit d80e66e). Skip clone if the
        # directory is already populated, so we don't overwrite patches.
        info "$name already populated at $target (vendored in ROSE), skipping clone"
    else
        info "Cloning $name → $target"
        git clone "$url" "$target"
    fi
}

# ============================================================================
# Helper: download HuggingFace model
# ============================================================================
download_hf_model() {
    local repo_id="$1"
    local target="$2"
    local name
    name="$(basename "$target")"

    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        info "$name already downloaded, skipping"
    else
        info "Downloading $repo_id → $target"
        mkdir -p "$target"
        huggingface-cli download "$repo_id" --local-dir "$target"
    fi
}

# ============================================================================
# Helper: download single file
# ============================================================================
download_file() {
    local url="$1"
    local target="$2"

    if [ -f "$target" ]; then
        info "$(basename "$target") already exists, skipping"
    else
        info "Downloading $(basename "$target")"
        mkdir -p "$(dirname "$target")"
        wget -q --show-progress -O "$target" "$url"
    fi
}


# ############################################################################
#                          PART 1: CORE PIPELINE
# ############################################################################

if [ "$INSTALL_CORE" = true ]; then

info "============================================"
info "  STEP 1/4: Clone external code repositories"
info "============================================"

# --- SAM3 (Segment Anything Model 3) ---
clone_repo "https://github.com/facebookresearch/sam3" \
           "rose/vision/sam3"

# --- Depth Anything 3 ---
clone_repo "https://github.com/ByteDance-Seed/Depth-Anything-3" \
           "rose/vision/da3"

echo ""
info "============================================"
info "  STEP 2/4: Install editable packages"
info "============================================"

# DA3 requires pip install -e
if [ -f "rose/vision/da3/pyproject.toml" ] || [ -f "rose/vision/da3/setup.py" ]; then
    info "Installing DA3 (pip install -e --no-deps) ..."
    pip install -e rose/vision/da3 --no-deps 2>&1 | tail -1
else
    warn "DA3 pyproject.toml/setup.py not found, skipping pip install"
fi

# SAM3 also requires pip install -e (vendored with ROSE patches)
if [ -f "rose/vision/sam3/pyproject.toml" ] || [ -f "rose/vision/sam3/setup.py" ]; then
    info "Installing SAM3 (pip install -e --no-deps) ..."
    pip install -e rose/vision/sam3 --no-deps 2>&1 | tail -1
else
    warn "SAM3 pyproject.toml/setup.py not found, skipping pip install"
fi

echo ""
info "============================================"
info "  STEP 3/4: Download model weights"
info "============================================"

# --- SAM3 checkpoint (gated model, requires HF login) ---
# HuggingFace ID: facebook/sam3
download_hf_model "facebook/sam3" "rose/models/sam3"

# --- Depth Anything 3 (Nested Giant-Large, METRIC — the pipeline default) ---
# DA3Config.model_path = rose/models/da3nested-giant-large (require_metric=True).
# HuggingFace ID: depth-anything/DA3NESTED-Giant-Large
download_hf_model "depth-anything/DA3NESTED-Giant-Large" "rose/models/da3nested-giant-large"

# --- Qwen3-VL-4B-Instruct (dynamic-targets instance namer — the default) ---
# DynamicTargetsConfig.namer_model_path = rose/models/qwen3-vl-4b-instruct.
# Bake-off winner (close-up COCO 73.8%). Only needed when dynamic_targets.enabled.
download_hf_model "Qwen/Qwen3-VL-4B-Instruct" "rose/models/qwen3-vl-4b-instruct"

# --- Gemma-3-4B-IT (gated; OPTIONAL fallback namer, namer_provider=gemma) ---
# Accept license at: https://huggingface.co/google/gemma-3-4b-it
# Comment out if you only use the Qwen3-VL namer (the default).
download_hf_model "google/gemma-3-4b-it" "rose/models/gemma-3-4b-it"

# --- FastSAM-s ---
download_file \
    "https://huggingface.co/CASIA-IVA-Lab/FastSAM-s/resolve/main/FastSAM-s.pt" \
    "rose/models/fastsam/FastSAM-s.pt"

echo ""
info "============================================"
info "  STEP 4/4: Verify installation"
info "============================================"

MISSING=0
for d in rose/vision/sam3 rose/vision/da3 rose/vision/recognize-anything \
         rose/vision/Video-Depth-Anything rose/vision/sam2; do
    if [ -d "$d/.git" ]; then
        info "  ✓ $(basename $d)"
    else
        error "  ✗ $(basename $d) — missing"
        MISSING=$((MISSING + 1))
    fi
done

for f in rose/models/sam3/model.safetensors \
         rose/models/da3nested-giant-large/config.json \
         rose/models/qwen3-vl-4b-instruct/config.json \
         rose/models/fastsam/FastSAM-s.pt; do
    if [ -f "$f" ]; then
        info "  ✓ $(echo $f | sed 's|rose/models/||')"
    else
        error "  ✗ $(echo $f | sed 's|rose/models/||') — missing"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -eq 0 ]; then
    info "All core components installed successfully!"
else
    warn "$MISSING components missing. Check errors above."
fi

fi  # end INSTALL_CORE


# ############################################################################
#                            SUMMARY
# ############################################################################

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Model weights:  rose/models/"
echo "  Vision repos:   rose/vision/{sam3,da3,...}"
echo ""
echo "  Total disk usage:"
du -sh rose/models/ 2>/dev/null || true
echo "============================================"
