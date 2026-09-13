"""Optional engineering scorecard; not a primary scientific result.

The publication protocol uses Pareto frontiers and predefined operational
constraints instead of arbitrary metric weights. This file is retained only
for exploratory engineering dashboards and must be labelled accordingly.
"""
import argparse, json
from pathlib import Path


def rank(rows, recall_weight=.35, f1_weight=.25, speed_weight=.20, count_weight=.20):
    def norm(vals, value, reverse=False):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 1.0
        x = (value - lo) / (hi - lo)
        return 1 - x if reverse else x

    for group in sorted(set((r['scenario'], r['modality']) for r in rows)):
        items = [r for r in rows if (r['scenario'], r['modality']) == group]
        recalls = [r['recall'] for r in items]
        f1s = [r['f1'] for r in items]
        speeds = [r.get('fps') or 0 for r in items]
        errors = [r.get('count_abs_error', 0) for r in items]
        for r in items:
            r['engineering_score_only'] = (
                recall_weight * norm(recalls, r['recall'])
                + f1_weight * norm(f1s, r['f1'])
                + speed_weight * norm(speeds, r.get('fps') or 0)
                + count_weight * norm(errors, r.get('count_abs_error', 0), True)
            )
            r['scorecard_weights'] = {
                'recall': recall_weight, 'f1': f1_weight,
                'speed': speed_weight, 'count_error': count_weight,
            }
            r['result_type'] = 'engineering_weighted_score_only'
    return sorted(rows, key=lambda x: x['engineering_score_only'], reverse=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--output', default='scorecard.json')
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    rows = data if isinstance(data, list) else data.get('results', data.get('summary', []))
    Path(args.output).write_text(json.dumps(rank(rows), indent=2))
    print('Wrote optional engineering scorecard; do not use it as the primary paper conclusion.')


if __name__ == '__main__':
    main()
