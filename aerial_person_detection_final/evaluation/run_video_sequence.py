"""Run a detector over a recorded aerial video for systems demonstration.

This is not the primary accuracy benchmark. It produces raw frame detections
for a controlled video/sequence demonstration and can be passed to
simulate_uav_pipeline.py. It requires a trained YOLO checkpoint and OpenCV.
"""
import argparse
import json
from pathlib import Path

from models.adapters import YOLOv8Adapter, YOLO10Adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--model', choices=['yolov8n', 'yolov10n'], required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit('OpenCV is required for video sequence processing') from exc

    adapter_class = YOLOv8Adapter if args.model == 'yolov8n' else YOLO10Adapter
    detector = adapter_class(args.weights, ['person'], confidence=None, imgsz=args.imgsz)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f'Could not open video: {args.video}')
    records = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and frame_index >= args.max_frames):
            break
        predictions = detector.predict(frame, confidence=None)
        records.append({
            'image_id': f'frame-{frame_index:06d}',
            'frame_index': frame_index,
            'source_video': str(Path(args.video).resolve()),
            'scenario': 'simulated_uav_video',
            'modality': 'rgb',
            'result_type': 'real_rgb_video_sequence',
            'image_size': args.imgsz,
            'predictions': [prediction.__dict__ for prediction in predictions],
            'truths': [],
        })
        frame_index += 1
    capture.release()
    if not records:
        raise SystemExit('No frames were processed')
    Path(args.output).write_text(json.dumps(records, indent=2))
    print(f'Wrote {len(records)} video-frame detection records to {args.output}')


if __name__ == '__main__':
    main()
