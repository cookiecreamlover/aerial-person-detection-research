# Corrections applied after benchmark audits

## Focused person mapping

The publication repository contains only the VisDrone one-class study. The adapters map COCO `person` outputs and fine-tuned VisDrone `person` outputs to compact study class ID 0. Wildlife-specific class mappings are not part of this repository.

## Raw detections for AP

The model adapters retain raw detections and do not confidence-filter YOLO outputs during benchmark generation. AP50 and mAP50–95 are therefore computed from all available detections ranked by confidence. Precision, recall, F1, false positives per image, and missed-person rate are computed separately after applying the threshold selected on validation data.

## Locked threshold selection

`evaluation/select_threshold.py` evaluates candidate thresholds from 0.05 through 0.95, selects the validation threshold maximizing F1, resolves ties toward the higher threshold, and saves per-model metadata. Test evaluation requires this metadata and records `threshold_locked_before_test: true`.

## Explicit image size

All model adapters receive the declared `--imgsz` value. The primary experiment uses 1280 and the sensitivity experiment uses 640. This value is recorded in prediction and efficiency outputs.

## Evidence labels

Real RGB detector runs use `real_rgb_evaluation`. Synthetic results from earlier prototypes use `synthetic_stress_test`. These labels are preserved through scoring.
