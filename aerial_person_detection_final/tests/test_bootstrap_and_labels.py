import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.metrics import bootstrap_confidence_intervals


@dataclass
class Box:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    latency_ms: float = 0.0


def test_bootstrap_ci_shape():
    box = Box(0, 'person', 0.9, 0, 0, 10, 10)
    records = [{'predictions': [box], 'truths': [box]}]
    result = bootstrap_confidence_intervals(records, class_ids=[0], iterations=20, seed=7)
    assert result['map50_95']['lower'] == 1.0
    assert result['map50_95']['upper'] == 1.0
    assert result['map50_95']['unit'] == 'image_bootstrap'


def test_real_label_is_serializable():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'result.json'
        path.write_text(json.dumps({'result_type': 'real_rgb_evaluation'}))
        assert json.loads(path.read_text())['result_type'] == 'real_rgb_evaluation'


if __name__ == '__main__':
    test_bootstrap_ci_shape()
    test_real_label_is_serializable()
    print('Bootstrap and evidence-label tests passed.')
