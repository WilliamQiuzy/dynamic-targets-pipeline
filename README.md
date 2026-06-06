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

- NVIDIA GPU. Best on **Hopper (H100/H200)** — the prebuilt image bakes in
  Flash-Attention-3. Any CUDA GPU also works (falls back to FA2/SDPA).
- A Hugging Face account with access to the gated **`facebook/sam3`** checkpoint
  (accept the license on its HF model page), used by the tracker.

## Setup (Docker, recommended)

```bash
# 1) Build the image (on any amd64 Docker host; no GPU needed to build).
#    On a Hopper amd64 host this also compiles Flash-Attention-3 (~30 min).
docker build -t dynamic-targets .
#    On a non-amd64 host (e.g. Apple-Silicon Mac), skip the FA3 compile:
#    docker build --platform linux/amd64 --build-arg SKIP_FA3=1 -t dynamic-targets .

# 2) Run on a CUDA GPU host, mounting a local weights cache.
docker run --gpus all -it --rm -v $(pwd)/rose/models:/workspace/rose/rose/models dynamic-targets

# 3) Inside the container: log in to HF and download model weights (~one-time).
huggingface-cli login
bash scripts/setup.sh --core      # downloads SAM3, DA3 (metric), Qwen3-VL-4B, FastSAM
```

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
