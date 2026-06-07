# Dynamic-Targets Pipeline

Input a video → output a per-frame JSON describing every visible object with a
**9DOF oriented 3D box**, **2D pixel box**, **open-vocabulary instance name**, and
**absolute world-frame velocity**. Designed for indoor dynamic-object tracking from
a single monocular RGB video.

## Output schema (`ALL_FRAMES`)

```jsonc
{
  "metadata": { "num_frames", "num_instances", "image_width", "image_height",
                "source_fps", "coordinate_system", "field_guide" },
  "ALL_FRAMES": {
    "frame_0": {
      "timestamp": 0.0,                       // seconds from video start
      "visible_objects": [
        {
          "instance_id": 3,                   // unique, persistent across frames (tracking id)
          "instance_name": "office chair",    // open-vocabulary class
          "Ori_9DOF_corners": [[x,y,z], ...], // 8 world-frame corners of the oriented 3D box
          "Ori_9DOF_center":  [x, y, z],      // world-frame box center (metres)
          "Ori_9DOF_size":    [L, H, W],      // box size in metres (L,W horizontal; H vertical)
          "instance_pv_bbox": [x0, y0, x1, y1],// 2D pixel box [left, top, right, bottom]
          "absolute_velocity":[vx, vy, vz]     // world-frame velocity, m/s (camera-motion compensated)
        }
      ]
    },
    "frame_1": { ... }
  }
}
```

**Coordinate system** — world frame = camera at frame 0: `X` = right, `Y` = down
(gravity points +Y), `Z` = forward (into the scene). Units are **metres** (metric
monocular depth). Velocity is in the world frame, so it reflects true object motion
(camera ego-motion is already compensated).

## Requirements

- **Any NVIDIA CUDA GPU.** Flash-Attention is **OFF by default**, so the model
  runs on older cards too (e.g. V100) via PyTorch SDPA. On Hopper (H100/H200)
  set `config.sam3.enable_fa3 = True` (with FA3 compiled into the image) for max
  speed. To force FA off explicitly: env `ROSE_DISABLE_FA3=1`.
- A Hugging Face account with access to the gated **`facebook/sam3`** checkpoint
  (accept the license on its HF model page), used by the tracker.

## Setup (prebuilt image — no build needed)

A ready-to-run public image is on Docker Hub, so you do **not** need to build
anything — just pull and run:

```bash
# 1) Pull the prebuilt public image (amd64; runs on any CUDA GPU).
docker pull ziyueqiu/dynamic-targets:latest

# 2) Run on a CUDA GPU host. The mounted `models` dir caches weights across runs.
docker run --gpus all -it --rm \
  -v $(pwd)/models:/workspace/rose/rose/models \
  ziyueqiu/dynamic-targets:latest

# 3) Inside the container: log in to HF, then ONE command downloads ALL models.
huggingface-cli login             # needed for gated facebook/sam3 + sam3.1 (accept their licenses first)
bash scripts/setup.sh --core      # downloads ALL weights: SAM3 + SAM3.1 + DA3 (metric) + Qwen3-VL-4B + FastSAM
                                  # (weights are NOT baked into the image; this fetches them once,
                                  #  into the mounted models dir so later runs reuse them)
```

<details>
<summary>Build the image yourself instead (optional)</summary>

```bash
# On any amd64 Docker host; no GPU needed to build. Hopper amd64 also compiles FA3 (~30 min).
docker build -t dynamic-targets .
# On a non-amd64 host (e.g. Apple-Silicon Mac), skip the FA3 compile:
docker buildx build --platform linux/amd64 --build-arg SKIP_FA3=1 -t dynamic-targets .
```
</details>

## Run

```bash
python scripts/run_dynamic_targets.py path/to/video.mp4 -o out.json
#   --no-name        skip VLM naming (instance_name = "object"), faster
#   --only-dynamic   emit only moving objects
```

The pipeline writes the `ALL_FRAMES` JSON to `out.json`.

## Notes

- Instance naming uses a local **Qwen3-VL-4B** vision-language model
  (`rose/models/qwen3-vl-4b-instruct`, open-vocabulary).
- 3D boxes/velocity come from monocular metric depth (DepthAnything-3) + per-frame
  camera pose; centre & velocity are produced by a constant-velocity RTS Kalman
  smoother (robust to depth outliers). Far objects (>10 m) have larger depth noise.
- Models are **not** baked into the image — `scripts/setup.sh --core` downloads them.
