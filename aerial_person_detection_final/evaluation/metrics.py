"""Detection metrics and image-bootstrap confidence intervals."""
import math
import random


def box_iou(a, b):
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    ab = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    return inter / (aa + ab - inter) if aa + ab - inter else 0.0


def match(predictions, truths, threshold=.5):
    pairs = []
    for pi, p in enumerate(predictions):
        for ti, t in enumerate(truths):
            if p.class_id == t.class_id:
                pairs.append((box_iou(p, t), pi, ti))
    used_p, used_t = set(), set()
    tp = 0
    for score, pi, ti in sorted(pairs, reverse=True):
        if score >= threshold and pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
    return tp, len(predictions) - tp, len(truths) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, 2 * p * r / (p + r) if p + r else 0


def average_precision(records, class_id, iou_threshold=.5):
    total_gt = sum(1 for r in records for t in r['truths'] if t.class_id == class_id)
    if total_gt == 0:
        return None
    entries = []
    for image_index, r in enumerate(records):
        for p in r['predictions']:
            if p.class_id == class_id:
                entries.append((p.confidence, image_index, p))
    entries.sort(key=lambda x: x[0], reverse=True)
    used_by_image = {}
    tp, fp = [], []
    for _, image_index, p in entries:
        truths = [t for t in records[image_index]['truths'] if t.class_id == class_id]
        used = used_by_image.setdefault(image_index, set())
        best = (-1, None)
        for i, t in enumerate(truths):
            if i not in used:
                score = box_iou(p, t)
                if score > best[0]:
                    best = (score, i)
        if best[0] >= iou_threshold:
            tp.append(1)
            fp.append(0)
            used.add(best[1])
        else:
            tp.append(0)
            fp.append(1)
    if not tp:
        return 0.0
    cum_tp, cum_fp, a, b = [], [], 0, 0
    for x, y in zip(tp, fp):
        a += x
        b += y
        cum_tp.append(a)
        cum_fp.append(b)
    recalls = [x / total_gt for x in cum_tp]
    precisions = [x / (x + y) for x, y in zip(cum_tp, cum_fp)]
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1] for i in range(len(mrec) - 1))


def summarize(records, class_ids=None, iou_threshold=.5):
    tp = fp = fn = 0
    lat = []
    gt = pred = 0
    for rec in records:
        a, b, c = match(rec['predictions'], rec['truths'], iou_threshold)
        tp += a
        fp += b
        fn += c
        gt += len(rec['truths'])
        pred += len(rec['predictions'])
        lat.extend(p.latency_ms for p in rec['predictions'] if p.latency_ms)
    p, r, f = prf(tp, fp, fn)
    class_ids = class_ids or sorted(set(t.class_id for rec in records for t in rec['truths']))
    ap50s = [average_precision(records, c, .5) for c in class_ids]
    map50s = [x for x in ap50s if x is not None]
    maps = []
    for threshold in [x / 100 for x in range(50, 100, 5)]:
        vals = [average_precision(records, c, threshold) for c in class_ids]
        vals = [x for x in vals if x is not None]
        if vals:
            maps.append(sum(vals) / len(vals))
    mean_latency = sum(lat) / len(lat) if lat else None
    return {
        'images': len(records), 'ground_truths': gt, 'predictions': pred,
        'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r, 'f1': f,
        'ap50': sum(map50s) / len(map50s) if map50s else None,
        'map50_95': sum(maps) / len(maps) if maps else None,
        'mean_latency_ms': mean_latency,
        'fps': 1000 / mean_latency if mean_latency else None,
        'iou_threshold': iou_threshold,
        'false_positives_per_image': fp / len(records) if records else None,
        'missed_person_rate': fn / gt if gt else None,
        'per_class_ap50': {str(c): average_precision(records, c, .5) for c in class_ids},
    }


def _percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(q * len(values)))]


def bootstrap_confidence_intervals(records, class_ids=None, iterations=2000, seed=2026):
    """Return percentile 95% CIs from image-level bootstrap resampling.

    Resampling is performed at image level, preserving all predictions and
    truths belonging to a sampled image. This estimates test-image sampling
    uncertainty and is not a substitute for variation across training seeds.
    """
    if not records:
        raise ValueError('At least one record is required for bootstrap CIs')
    rng = random.Random(seed)
    values = {name: [] for name in ('precision', 'recall', 'f1', 'ap50', 'map50_95')}
    for _ in range(iterations):
        sample = [records[rng.randrange(len(records))] for _ in records]
        score = summarize(sample, class_ids=class_ids)
        for name in values:
            if score[name] is not None:
                values[name].append(score[name])
    return {
        name: {'lower': _percentile(samples, .025), 'upper': _percentile(samples, .975), 'iterations': iterations, 'unit': 'image_bootstrap'}
        for name, samples in values.items()
    }


def count_error(predicted, truth):
    return abs(predicted - truth), (predicted - truth) / truth if truth else None


def rmse(errors):
    return math.sqrt(sum(x * x for x in errors) / len(errors)) if errors else None
