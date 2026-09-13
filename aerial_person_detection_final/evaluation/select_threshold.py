"""Select validation F1 thresholds from raw prediction records."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from models.adapters import Prediction
from evaluation.thresholds import select_f1_threshold


def convert(record):
    result = dict(record)
    result['predictions'] = [Prediction(**{k: item[k] for k in ('class_id', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2', 'latency_ms')}) for item in record['predictions']]
    result['truths'] = [Prediction(x['class_id'], x['label'], x.get('confidence', 1.0), x['x1'], x['y1'], x['x2'], x['y2'], 0.0) for x in record['truths']]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [convert(row) for row in json.loads(Path(args.input).read_text())]
    groups = defaultdict(list)
    for row in rows:
        groups[row['model']].append(row)
    output = {model: select_f1_threshold(records) for model, records in groups.items()}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
