"""Cross-check custom AP/mAP against pycocotools on the same records.

This utility is a validation aid, not the primary scorer. It exits with a
clear installation message when pycocotools is unavailable.
"""
import argparse
import json
from pathlib import Path

from evaluation.metrics import summarize
from evaluation.score_predictions import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Raw prediction JSON for one model')
    args = parser.parse_args()
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit('Install pycocotools to run the trusted COCOeval cross-check.') from exc

    raw = json.loads(Path(args.input).read_text())
    records = [convert(row) for row in raw]
    images = [{'id': i} for i in range(len(records))]
    annotations = []
    predictions = []
    annotation_id = 1
    for image_id, record in enumerate(records):
        for truth in record['truths']:
            annotations.append({'id': annotation_id, 'image_id': image_id, 'category_id': 1, 'bbox': [truth.x1, truth.y1, truth.x2 - truth.x1, truth.y2 - truth.y1], 'area': max(0, truth.x2 - truth.x1) * max(0, truth.y2 - truth.y1), 'iscrowd': 0})
            annotation_id += 1
        for prediction in record['predictions']:
            predictions.append({'image_id': image_id, 'category_id': 1, 'bbox': [prediction.x1, prediction.y1, prediction.x2 - prediction.x1, prediction.y2 - prediction.y1], 'score': prediction.confidence})
    dataset = {'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'person'}]}
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    detections = coco.loadRes(predictions)
    evaluation = COCOeval(coco, detections, 'bbox')
    evaluation.params.imgIds = list(range(len(records)))
    evaluation.params.catIds = [1]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    custom = summarize(records, class_ids=[0])
    print(json.dumps({'custom_ap50': custom['ap50'], 'custom_map50_95': custom['map50_95'], 'cocoeval_ap50': float(evaluation.stats[1]) if evaluation.stats is not None else None, 'cocoeval_map50_95': float(evaluation.stats[0]) if evaluation.stats is not None else None}, indent=2))


if __name__ == '__main__':
    main()
