from dataclasses import dataclass

from evaluation.metrics import average_precision
from evaluation.thresholds import filter_records, select_f1_threshold
from models.adapters import map_coco_id


@dataclass
class B:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    latency_ms: float = 0.0


def run():
    assert map_coco_id(0, ['person']) == (0, 'person')
    assert map_coco_id(1, ['person']) == (0, 'person')
    assert map_coco_id(15, ['person']) is None

    truth = B(0, 'person', 1, 0, 0, 10, 10)
    high = B(0, 'person', .9, 0, 0, 10, 10)
    low = B(0, 'person', .1, 0, 0, 10, 10)
    records = [{'predictions': [high, low], 'truths': [truth]}]
    assert average_precision(records, 0, .5) == 1.0

    selected = select_f1_threshold(records, thresholds=[.05, .5, .95])
    assert selected['threshold_selection_metric'] == 'F1'
    assert selected['threshold_locked_before_test'] is True
    assert selected['validation_selected_threshold'] == .5
    assert len(filter_records(records, .5)[0]['predictions']) == 1
    print('Review-fix tests passed: person mapping, raw AP, and locked threshold.')


if __name__ == '__main__':
    run()
