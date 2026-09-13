"""Score raw predictions with standard AP and locked operating metrics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from models.adapters import Prediction
from evaluation.metrics import bootstrap_confidence_intervals, summarize
from evaluation.thresholds import filter_records


def load_prediction(x):
    return Prediction(x['class_id'], x['label'], x['confidence'], x['x1'], x['y1'], x['x2'], x['y2'], x.get('latency_ms', 0))


def convert(row):
    row = dict(row)
    row['predictions'] = [load_prediction(x) for x in row['predictions']]
    row['truths'] = [load_prediction({**x, 'confidence': 1, 'latency_ms': 0}) for x in row['truths']]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', default='scored_results.json')
    parser.add_argument('--bootstrap-iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    rows = [convert(row) for row in json.loads(Path(args.input).read_text())]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row.get('result_type', 'unknown'))].append(row)

    output = []
    for (model, result_type), records in groups.items():
        thresholds = [r.get('validation_selected_threshold') for r in records]
        if any(t is None for t in thresholds):
            raise SystemExit(f'Missing locked validation threshold for {model}')
        threshold = thresholds[0]
        if any(t != threshold for t in thresholds):
            raise SystemExit(f'Inconsistent locked thresholds for {model}')

        raw_score = summarize(records, class_ids=[0])
        operating_records = filter_records(records, threshold)
        operating_score = summarize(operating_records, class_ids=[0])
        ci = bootstrap_confidence_intervals(operating_records, class_ids=[0], iterations=args.bootstrap_iterations, seed=args.seed)
        output.append({
            'model': model,
            'result_type': result_type,
            'images': raw_score['images'],
            'ground_truths': raw_score['ground_truths'],
            'predictions': raw_score['predictions'],
            'ap50': raw_score['ap50'],
            'map50_95': raw_score['map50_95'],
            'precision': operating_score['precision'],
            'recall': operating_score['recall'],
            'f1': operating_score['f1'],
            'false_positives_per_image': operating_score['false_positives_per_image'],
            'missed_person_rate': operating_score['missed_person_rate'],
            'validation_selected_threshold': threshold,
            'threshold_selection_metric': 'F1',
            'threshold_locked_before_test': all(r.get('threshold_locked_before_test', False) for r in records),
            'bootstrap_95_ci': ci,
            'ap_prediction_scope': 'all_raw_detections_ranked_by_confidence',
            'operating_point_scope': 'detections_at_locked_validation_threshold',
        })
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
