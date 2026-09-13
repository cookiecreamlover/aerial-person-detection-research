"""Common detector API for the one-class VisDrone person study.

Adapters return raw detections by default. Confidence filtering is applied in
the evaluation layer so AP/mAP can use the full confidence-ranked list.
"""
from dataclasses import dataclass
from time import perf_counter

COCO_ID_TO_NAME = {0: 'person', 1: 'person'}


@dataclass
class Prediction:
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    latency_ms: float = 0.0


def map_coco_id(raw_id, study_classes):
    """Map common COCO person IDs to the compact one-class study ID."""
    name = COCO_ID_TO_NAME.get(int(raw_id))
    if name is None or name not in study_classes:
        return None
    return study_classes.index(name), name


class UltralyticsAdapter:
    def __init__(self, weights, model_name, class_names, confidence=None, imgsz=1280):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Ultralytics is required for real YOLO evaluation.') from exc
        self.model = YOLO(weights)
        self.name = model_name
        self.class_names = class_names
        self.confidence = confidence
        self.imgsz = imgsz

    def predict(self, image, confidence=None):
        threshold = self.confidence if confidence is None else confidence
        # conf=0.0 retains the full ranked list for standard AP calculation.
        model_confidence = 0.0 if threshold is None else threshold
        started = perf_counter()
        result = self.model.predict(image, conf=model_confidence, imgsz=self.imgsz, verbose=False)[0]
        elapsed = (perf_counter() - started) * 1000
        raw_names = getattr(result, 'names', getattr(self.model, 'names', {}))
        output = []
        for box in result.boxes:
            raw_id = int(box.cls.item())
            raw_name = raw_names.get(raw_id, COCO_ID_TO_NAME.get(raw_id, '')) if isinstance(raw_names, dict) else COCO_ID_TO_NAME.get(raw_id, '')
            if raw_name not in self.class_names:
                continue
            confidence_value = float(box.conf.item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            output.append(Prediction(self.class_names.index(raw_name), raw_name, confidence_value, x1, y1, x2, y2, elapsed))
        return output


class YOLOv8Adapter(UltralyticsAdapter):
    def __init__(self, weights='yolov8n.pt', class_names=None, confidence=None, imgsz=1280):
        super().__init__(weights, 'YOLOv8n', class_names or ['person'], confidence, imgsz)


class YOLO10Adapter(UltralyticsAdapter):
    def __init__(self, weights='yolov10n.pt', class_names=None, confidence=None, imgsz=1280):
        super().__init__(weights, 'YOLOv10n', class_names or ['person'], confidence, imgsz)


class FasterRCNNAdapter:
    def __init__(self, class_names=None, confidence=None, weights='DEFAULT', checkpoint=None, imgsz=1280):
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
        except ImportError as exc:
            raise RuntimeError('PyTorch and Torchvision are required for Faster R-CNN real evaluation.') from exc
        self.torch = torch
        self.name = 'Faster R-CNN R50-FPN'
        self.class_names = class_names or ['person']
        self.confidence = confidence
        self.checkpoint = checkpoint
        self.imgsz = imgsz
        if checkpoint:
            self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=len(self.class_names) + 1)
            state = self.torch.load(checkpoint, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.model.load_state_dict(state)
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
        else:
            pretrained_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if weights == 'DEFAULT' else weights
            self.model = fasterrcnn_resnet50_fpn(weights=pretrained_weights)
            self.transforms = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()
        self.model.eval()

    def predict(self, image, confidence=None):
        threshold = self.confidence if confidence is None else confidence
        started = perf_counter()
        tensor = self.transforms(image)
        with self.torch.no_grad():
            result = self.model([tensor])[0]
        elapsed = (perf_counter() - started) * 1000
        output = []
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            confidence_value = float(score)
            if threshold is not None and confidence_value < threshold:
                continue
            raw_label = int(label)
            if self.checkpoint and 1 <= raw_label <= len(self.class_names):
                mapped = (raw_label - 1, self.class_names[raw_label - 1])
            else:
                mapped = map_coco_id(raw_label, self.class_names)
            if mapped is None:
                continue
            class_id, name = mapped
            x1, y1, x2, y2 = map(float, box.tolist())
            output.append(Prediction(class_id, name, confidence_value, x1, y1, x2, y2, elapsed))
        return output
