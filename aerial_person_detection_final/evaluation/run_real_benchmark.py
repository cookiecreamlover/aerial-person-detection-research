"""Run all detectors on a fixed RGB JSONL manifest.

The runner supports two phases:
1. ``--split val`` writes raw detections for threshold selection.
2. ``--split test`` consumes a threshold JSON produced from validation.

Raw detections are never confidence-clipped in the runner, allowing standard
AP/mAP calculation from the complete ranked detection list. Thresholding is
performed only for P/R/F1 operating-point metrics.
"""
import argparse
import json
from pathlib import Path

from models.adapters import YOLOv8Adapter, YOLO10Adapter, FasterRCNNAdapter


def build_models(args):
    classes = ['person']
    return [
        YOLOv8Adapter(args.yolov8, classes, confidence=None, imgsz=args.imgsz),
        YOLO10Adapter(args.yolo10, classes, confidence=None, imgsz=args.imgsz),
        FasterRCNNAdapter(classes, confidence=None, checkpoint=args.faster_rcnn_checkpoint, imgsz=args.imgsz),
    ]


def load_rows(path, scenario='survivor', modality='rgb'):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get('scenario') == scenario and row.get('modality') == modality]
    if not selected:
        raise SystemExit('No rows match the requested survivor/rgb manifest selection.')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', required=True, choices=['val', 'test'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--yolov8', required=True)
    parser.add_argument('--yolo10', required=True)
    parser.add_argument('--faster-rcnn-checkpoint', required=True)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--thresholds', default=None, help='JSON from select_threshold.py; required for --split test')
    args = parser.parse_args()
    if args.split == 'test' and not args.thresholds:
        raise SystemExit('--thresholds is required for locked test evaluation')

    rows = load_rows(args.manifest)
    locked = json.loads(Path(args.thresholds).read_text()) if args.thresholds else {}
    models = build_models(args)
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit('OpenCV is required to load benchmark images.') from exc

    output = []
    for model in models:
        threshold_metadata = locked.get(model.name, {})
        if args.split == 'test' and not threshold_metadata.get('threshold_locked_before_test'):
            raise SystemExit(f'Missing locked threshold metadata for {model.name}')
        threshold = threshold_metadata.get('validation_selected_threshold')
        for row in rows:
            image = cv2.imread(row['image'])
            if image is None:
                raise FileNotFoundError(row['image'])
            predictions = model.predict(image, confidence=None)
            output.append({
                'image_id': row.get('image_id', row['image']),
                'model': model.name,
                'split': args.split,
                'scenario': 'survivor',
                'modality': 'rgb',
                'result_type': 'real_rgb_evaluation',
                'image_size': args.imgsz,
                'predictions': [p.__dict__ for p in predictions],
                'truths': row.get('truths', []),
                'validation_selected_threshold': threshold,
                'threshold_selection_metric': threshold_metadata.get('threshold_selection_metric', 'F1'),
                'threshold_locked_before_test': bool(threshold_metadata.get('threshold_locked_before_test', False)),
            })
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f'Wrote {len(output)} raw model-image records to {args.output}')


if __name__ == '__main__':
    main()
