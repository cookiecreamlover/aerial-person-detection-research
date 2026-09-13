"""Run a systems demonstration over recorded aerial frames.

This is not detector evaluation and does not infer survivor ground truth. It
consumes saved raw detections, performs simple temporal candidate association,
creates
candidate IDs, and emits map/alert-ready records. Coordinates are copied only
when supplied by the input manifest; otherwise they remain unavailable.
"""
import argparse
import json
import math
from pathlib import Path


def center(prediction):
    return ((prediction['x1'] + prediction['x2']) / 2, (prediction['y1'] + prediction['y2']) / 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def run(records, association_radius=80.0):
    tracks = []
    next_id = 1
    output = []
    for frame_index, record in enumerate(records):
        frame_candidates = []
        for prediction in record.get('predictions', []):
            point = center(prediction)
            if prediction.get('validation_selected_threshold') is not None and prediction['confidence'] < prediction['validation_selected_threshold']:
                continue
            match = next((track for track in tracks if distance(track['last_center'], point) <= association_radius), None)
            if match is None:
                match = {'candidate_id': f'CAND-{next_id:04d}', 'last_center': point, 'frames': 0}
                tracks.append(match)
                next_id += 1
            match['last_center'] = point
            match['frames'] += 1
            frame_candidates.append({
                'candidate_id': match['candidate_id'],
                'confidence': prediction['confidence'],
                'bbox': [prediction['x1'], prediction['y1'], prediction['x2'], prediction['y2']],
                'coordinate': record.get('coordinate'),
            })
        output.append({
            'frame_index': frame_index,
            'image_id': record.get('image_id'),
            'candidate_persons': frame_candidates,
            'pipeline_status': 'simulated_uav_frame_processed',
            'physical_uav_deployment': False,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Raw test prediction JSON')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--association-radius', type=float, default=80.0, help='Pixel radius for simple temporal candidate association')
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text())
    if args.model:
        records = [record for record in records if record.get('model') == args.model]
    if not records:
        raise SystemExit('No prediction records selected')
    result = {
        'result_type': 'simulated_uav_pipeline_demonstration',
        'association_method': 'simple_temporal_candidate_association',
        'physical_uav_deployment': False,
        'ground_truth_claim': False,
        'source_result_type': records[0].get('result_type'),
        'frames': run(records, args.association_radius),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"Wrote simulated UAV pipeline output to {args.output}")


if __name__ == '__main__':
    main()
