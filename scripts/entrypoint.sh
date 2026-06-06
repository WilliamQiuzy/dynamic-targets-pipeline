#!/usr/bin/env bash
# ROSE container entrypoint.
# - Optionally starts SSH (if PUBLIC_KEY or ROOT_PASSWORD is set — RunPod convention).
# - Warns if model weights are missing under rose/models/.
# - Then exec()s whatever was passed as CMD (default: bash).

set -e

# ── Optional SSH ──────────────────────────────────────────────
if [ -n "${PUBLIC_KEY:-}" ] || [ -n "${ROOT_PASSWORD:-}" ]; then
    service ssh start >/dev/null 2>&1 || true
    if [ -n "${ROOT_PASSWORD:-}" ]; then
        echo "root:${ROOT_PASSWORD}" | chpasswd
    fi
    if [ -n "${PUBLIC_KEY:-}" ]; then
        mkdir -p /root/.ssh
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
fi

# ── Weight check ──────────────────────────────────────────────
missing=0
for d in da3-small sam3 fastsam gemma-3-4b-it; do
    if ! find "rose/models/$d" -mindepth 1 -not -name '.gitkeep' -not -name 'README.md' \
        2>/dev/null | grep -q .; then
        missing=1
        break
    fi
done

if [ "$missing" = "1" ]; then
    cat <<EOF >&2
[ROSE] Model weights not detected under rose/models/.
       Either:
         (a) re-run with -v \$(pwd)/rose/models:/workspace/rose/rose/models
             to mount your existing weights, OR
         (b) inside the container, run:
                 huggingface-cli login   # for gated models (Gemma, SAM3)
                 bash scripts/setup.sh --core
EOF
fi

exec "$@"
