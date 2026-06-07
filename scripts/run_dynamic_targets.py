"""CLI: input video → dynamic-targets ALL_FRAMES JSON (动态目标管线).

Usage:
    python scripts/run_dynamic_targets.py <video.mp4> [-o out.json] [--no-name]

Runs the ROSE pipeline with the dynamic-targets export enabled and writes the
client schema (per-frame visible objects with 9DOF oriented boxes, 2D pixel
boxes, open-vocabulary instance names, and absolute world-frame velocity).
"""
import os, sys, json, time, argparse
# Reduce CUDA fragmentation (helps memory-tight GPUs like V100/T4 avoid OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Repo root = parent of this scripts/ dir, so it works from ANY install location.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".ultralytics"))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "rose/vision/sam3"))
from rose.engine.config.rose_config import ROSEConfig
from rose.engine.server.warm_server import WarmModelPool, InferenceRequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default=None, help="output JSON path (default: <video>.dynamic_targets.json)")
    ap.add_argument("--no-name", action="store_true", help="skip VLM naming (instance_name='object')")
    ap.add_argument("--only-dynamic", action="store_true", help="emit only moving objects")
    args = ap.parse_args()

    cfg = ROSEConfig()
    cfg.dynamic_targets.enabled = True
    cfg.dynamic_targets.name_objects = not args.no_name
    cfg.dynamic_targets.only_dynamic = args.only_dynamic

    pool = WarmModelPool(cfg)
    t = time.time(); pool.load_all(); pool.warmup_cuda(); pool._status = "ready"
    print(f"[models ready in {time.time()-t:.1f}s]", flush=True)

    t = time.time()
    r = pool.run_inference(InferenceRequest(video_path=args.video))
    if r.status != "ok" or r.dynamic_targets is None:
        print(f"FAILED: status={r.status} err={r.error_message}", flush=True)
        sys.exit(1)
    dt = r.dynamic_targets
    out = args.out or (os.path.splitext(args.video)[0] + ".dynamic_targets.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dt, f, ensure_ascii=False, indent=2)
    md = dt["metadata"]
    print(f"[done in {time.time()-t:.1f}s] {md['num_instances']} instances over "
          f"{md['num_frames']} frames → {out}", flush=True)


if __name__ == "__main__":
    main()
