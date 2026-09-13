from pathlib import Path

from dataclasses import dataclass
from evaluation.metrics import average_precision


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


def test_low_confidence_detection_contributes_to_ap():
    truth = Box(0, 'person', 1.0, 0, 0, 10, 10)
    detections = [
        Box(0, 'person', .9, 0, 0, 10, 10),
        Box(0, 'person', .1, 0, 0, 10, 10),
    ]
    assert average_precision([{'predictions': detections, 'truths': [truth]}], 0) == 1.0


def test_image_size_is_explicit_in_adapters_and_benchmark():
    adapter_source = Path('models/adapters.py').read_text()
    runner_source = Path('evaluation/run_real_benchmark.py').read_text()
    efficiency_source = Path('evaluation/benchmark_efficiency.py').read_text()
    assert 'imgsz=self.imgsz' in adapter_source
    assert "parser.add_argument('--imgsz', type=int, default=1280)" in runner_source
    assert 'parser.add_argument("--imgsz", type=int, default=1280)' in efficiency_source


if __name__ == '__main__':
    test_low_confidence_detection_contributes_to_ap()
    test_image_size_is_explicit_in_adapters_and_benchmark()
    print('Protocol-control tests passed.')
