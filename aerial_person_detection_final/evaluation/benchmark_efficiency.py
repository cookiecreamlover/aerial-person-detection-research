"""Measure deployment efficiency on a fixed image manifest.

The script reports measured end-to-end inference timing only. It does not use
synthetic runtime placeholders. Component-level preprocessing and
postprocessing timings are not claimed because the three frameworks expose
those stages through different APIs.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from models.adapters import YOLOv8Adapter, YOLO10Adapter, FasterRCNNAdapter


def file_size_mb(path):
    return Path(path).stat().st_size / (1024 * 1024) if path and Path(path).exists() else None


def torch_module_stats(model):
    module = getattr(model, "model", None)
    if module is None:
        return {"parameter_count": None, "trainable_parameter_count": None}
    try:
        parameters = list(module.parameters())
        return {
            "parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
        }
    except (AttributeError, TypeError):
        return {"parameter_count": None, "trainable_parameter_count": None}


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def read_peak_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--yolov8", required=True)
    parser.add_argument("--yolo10", required=True)
    parser.add_argument("--faster-rcnn-checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--environment", default=None, help="Captured environment JSON to include in output")
    parser.add_argument("--min-timed-samples", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required") from exc

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    rows = rows[:args.limit] if args.limit else rows
    classes = ["person"]
    models = [
        (YOLOv8Adapter(args.yolov8, classes, confidence=None, imgsz=args.imgsz), args.yolov8),
        (YOLO10Adapter(args.yolo10, classes, confidence=None, imgsz=args.imgsz), args.yolo10),
        (FasterRCNNAdapter(classes, confidence=None, checkpoint=args.faster_rcnn_checkpoint, imgsz=args.imgsz), args.faster_rcnn_checkpoint),
    ]
    images = []
    for row in rows:
        image = cv2.imread(row["image"])
        if image is None:
            raise FileNotFoundError(row["image"])
        images.append(image)
    if not images:
        raise SystemExit("The manifest contains no images")
    if len(images) * args.repeats < args.min_timed_samples:
        raise SystemExit(f"Need at least {args.min_timed_samples} timed samples; increase --repeats or manifest size")
    environment = None
    if args.environment:
        environment = json.loads(Path(args.environment).read_text())

    output = []
    for model, checkpoint in models:
        for image in images[:min(args.warmup, len(images))]:
            model.predict(image)
        cuda_sync()
        reset_gpu_memory()
        durations = []
        for _ in range(args.repeats):
            for image in images:
                cuda_sync()
                start = time.perf_counter()
                model.predict(image)
                cuda_sync()
                durations.append((time.perf_counter() - start) * 1000)
        durations.sort()
        mean = statistics.mean(durations)
        output.append({
            "model": model.name,
            "images": len(images),
            "image_size": args.imgsz,
            "repeats": args.repeats,
            "warmup_images": min(args.warmup, len(images)),
            "mean_end_to_end_latency_ms": mean,
            "std_end_to_end_latency_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "median_end_to_end_latency_ms": statistics.median(durations),
            "p95_end_to_end_latency_ms": durations[min(len(durations) - 1, int(0.95 * len(durations)))],
            "throughput_images_per_second": 1000 / mean if mean else None,
            "checkpoint_size_mb": file_size_mb(checkpoint),
            "peak_gpu_memory_mb": read_peak_gpu_memory(),
            "environment_file": args.environment,
            "environment": environment,
            **torch_module_stats(model),
            "timing_scope": "end_to_end_model_call_including_framework_preprocessing_and_postprocessing",
            "measurement_type": "measured_inference",
        })
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote measured efficiency results to {args.output}")


if __name__ == "__main__":
    main()
