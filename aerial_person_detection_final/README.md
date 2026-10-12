Aerial Person Detection for Drone-Based Survivor Search

This repository contains the focused publication study comparing YOLOv8n, YOLOv10n, and Faster R-CNN ResNet-50-FPN on RGB aerial-person detection. The paper evaluates feasibility under controlled image and computational constraints; it does not claim deployment on a physical drone.

A detected person is a rescue candidate, not a confirmed survivor. The primary study uses VisDrone2019-DET and merges its pedestrian and people classes into one person class. Thermal sensing, tracking, GPS accuracy, and autonomous flight are outside the primary paper.

Evidence layers

Layer
Evidence
Claim supported
Aerial detection
Real labelled VisDrone images
Detector quality on aerial imagery
Hardware feasibility
Benchmark on a declared target computer
Computational feasibility for possible embedded use
UAV pipeline
Recorded aerial sequence with simulated flight path
Integration demonstration only




The final paper must not say that a model achieved a particular frame rate on our drone, because no physical UAV deployment is claimed. It may say that the model was measured on the declared target hardware and is or is not computationally feasible for a possible onboard pipeline.

Study pipeline

Plain Text


VisDrone2019-DET → one-class person conversion → train three models
→ select threshold on validation → lock threshold → test evaluation
→ accuracy + hardware feasibility + bootstrap CI → Pareto analysis
→ optional simulated UAV pipeline demonstration



Repository layout

Directory
Purpose
configs/
Frozen experimental settings
data/
Dataset conversion scripts
evaluation/
Prediction, scoring, confidence intervals, and efficiency
models/
Common model adapters
analysis/
Pareto visualization and optional engineering scorecard
docs/
Full study protocol
tests/
Regression tests
scripts/
Reproducibility helpers
results/
Generated outputs; initially empty




Run checks

Bash


python3 scripts/run_checks.py



Prepare VisDrone

Download the official VisDrone2019-DET train, validation, and test-dev archives from the 

VisDrone repository, then run the converter for each official split:

Bash


PYTHONPATH=. python3 data/prepare_visdrone_person.py \
  --root /path/to/VisDrone2019-DET \
  --split test-dev \
  --output-root data/visdrone_person \
  --manifest data/visdrone_person/test-dev.jsonl



The converter retains empty images so false positives are measured correctly.

The study uses test-dev only if its ground-truth annotations are available locally for the evaluator. If the downloaded archive does not include usable test-dev annotations, stop before training and define a sequence-aware held-out split from the annotated train/validation material. Do not silently evaluate on the validation set after threshold selection.

Train the three models

The original repository contains evaluation adapters but not a training entry point. The reproducible training scripts added here are:

•
scripts/train_yolo.py for YOLOv8n and YOLOv10n.

•
scripts/train_fasterrcnn.py for one-class Faster R-CNN ResNet-50-FPN.

Install the dependencies in a CUDA-enabled environment when possible, then create configs/visdrone_person.yaml from the example and replace its path with the absolute path to data/visdrone_person:

Bash


cp configs/visdrone_person.yaml.example configs/visdrone_person.yaml
# edit configs/visdrone_person.yaml and set path: /absolute/path/to/data/visdrone_person



Convert all annotated splits before training:

Bash


for split in train val test-dev; do
  PYTHONPATH=. python3 data/prepare_visdrone_person.py \
    --root /path/to/VisDrone2019-DET \
    --split "$split" \
    --output-root data/visdrone_person \
    --manifest "data/visdrone_person/$split.jsonl"
done



Train the two YOLO baselines with the same seed, input size, and nominal epoch budget:

Bash


python3 scripts/train_yolo.py --model yolov8n.pt --name yolov8n_person \
  --data configs/visdrone_person.yaml --epochs 100 --imgsz 1280 --device 0
python3 scripts/train_yolo.py --model yolov10n.pt --name yolov10n_person \
  --data configs/visdrone_person.yaml --epochs 100 --imgsz 1280 --device 0



Train Faster R-CNN using the JSONL manifest. Its output is a plain state_dict, which is the checkpoint format expected by models/adapters.py:

Bash


python3 scripts/train_fasterrcnn.py \
  --manifest data/visdrone_person/train.jsonl \
  --output results/training/fasterrcnn_person.pt \
  --epochs 20 --batch 2 --device cuda



For a fair comparison, preserve the exact commands, seed, framework versions, device, and final checkpoint hashes. If memory is insufficient, lower --imgsz for YOLO or --batch for Faster R-CNN, record the change, and do not compare the altered run as if it used the primary 1280-pixel protocol.

Score predictions with confidence intervals

First run the raw benchmark on validation data, then select and save one F1-maximizing threshold per model:

Bash


PYTHONPATH=. python3 evaluation/run_real_benchmark.py \
  --manifest data/visdrone_person/val.jsonl --split val \
  --yolov8 /path/to/yolov8n_person.pt --yolo10 /path/to/yolov10n_person.pt \
  --faster-rcnn-checkpoint /path/to/fasterrcnn_person.pt \
  --imgsz 1280 --output results/val_raw_predictions.json

PYTHONPATH=. python3 evaluation/select_threshold.py \
  results/val_raw_predictions.json --output results/locked_thresholds.json



Only after the thresholds are saved should the locked test run be performed:

Bash


PYTHONPATH=. python3 evaluation/run_real_benchmark.py \
  --manifest data/visdrone_person/test-dev.jsonl --split test \
  --thresholds results/locked_thresholds.json \
  --yolov8 /path/to/yolov8n_person.pt --yolo10 /path/to/yolov10n_person.pt \
  --faster-rcnn-checkpoint /path/to/fasterrcnn_person.pt \
  --imgsz 1280 --output results/test_raw_predictions.json



The runner stores all raw detections. AP50 and mAP50–95 are computed from the complete confidence-ranked list, while P/R/F1 are computed after applying the locked validation threshold.

Then score the test predictions:

Bash


PYTHONPATH=. python3 evaluation/score_predictions.py \
  results/test_raw_predictions.json \
  --output results/accuracy_results.json \
  --bootstrap-iterations 2000



The scorer preserves evidence labels such as real_rgb_evaluation and adds image-bootstrap 95% confidence intervals for precision, recall, F1, AP50, and mAP50–95.

Measure deployment efficiency

Bash


PYTHONPATH=. python3 evaluation/benchmark_efficiency.py \
  --manifest data/visdrone_person/test-dev.jsonl \
  --yolov8 /path/to/yolov8n_person.pt \
  --yolo10 /path/to/yolov10n_person.pt \
  --faster-rcnn-checkpoint /path/to/fasterrcnn_person.pt \
  --imgsz 1280 \
  --output results/efficiency_results.json



The timing scope is a complete framework model call, including framework preprocessing and postprocessing. The benchmark performs warm-up, repeated measurements, and CUDA synchronization when available. It reports mean, standard deviation, median, p95 latency, throughput, checkpoint size, parameter count, and peak GPU memory.

For publication timing, use the default 20 warm-up frames and 100 repeats, producing at least 100 timed samples. The preferred embedded target is a Raspberry Pi 4 8GB, if available; otherwise declare the exact substitute hardware. On a Pi 4, run the benchmark in CPU mode and record the captured environment. Do not describe the result as performance on a drone.

Capture the machine and software metadata before benchmarking:

Bash


python3 scripts/capture_environment.py



The resulting results/environment.json must accompany the latency results. Report CPU/GPU, RAM, CUDA, PyTorch, Torchvision, Ultralytics, image size, batch size, warm-up count, repeats, model-loading exclusion, disk-I/O exclusion, visualization exclusion, and GPU synchronization policy.

Simulated UAV pipeline

After the real test predictions exist, run the systems demonstration over the recorded aerial-image sequence:

Bash


PYTHONPATH=. python3 evaluation/simulate_uav_pipeline.py \
  results/test_raw_predictions.json \
  --model YOLOv8n \
  --output results/simulated_uav_pipeline.json



This demonstration performs candidate filtering, simple frame-to-frame association, candidate IDs, and optional coordinate passthrough. It is labelled simulated_uav_pipeline_demonstration, does not create survivor ground truth, and does not establish physical flight safety or drone deployment performance.

For a stronger systems figure, process a recorded aerial video frame by frame:

Bash


PYTHONPATH=. python3 evaluation/run_video_sequence.py \
  --video /path/to/recorded_aerial_sequence.mp4 \
  --model yolov8n \
  --weights /path/to/yolov8n_person.pt \
  --imgsz 1280 \
  --output results/video_raw_predictions.json

PYTHONPATH=. python3 evaluation/simulate_uav_pipeline.py \
  results/video_raw_predictions.json \
  --model YOLOv8n \
  --output results/video_uav_demo.json



The association method is intentionally named simple temporal candidate association. It is a systems demonstration, not a robust multi-object tracker. It should not be presented as reliable under crossings, camera motion, missed frames, or large scale changes.

Metric cross-check

The custom AP implementation can be compared with the recognized COCO API on identical raw predictions when pycocotools is installed:

Bash


PYTHONPATH=. python3 evaluation/crosscheck_cocoeval.py \
  results/test_raw_predictions.json



Record the custom and COCOeval values and investigate any material discrepancy before reporting results.

Scientific status

This repository provides a reproducible experimental framework. It does not contain VisDrone images, trained checkpoints, or real comparison results. Synthetic benchmark outputs from earlier prototypes must not be presented as detector accuracy or measured runtime.

The paper should use Pareto plots and the predefined operational rule. For real-time rescue, require recall of at least 0.80 and latency of at most 100 ms, then select the smallest feasible checkpoint. For accuracy-first mapping, select the highest mAP50–95 while reporting deployment costs.

Read the complete protocol in docs/survivor_detector_study_protocol.md.

References

[1] VisDrone dataset repository and benchmark description
[2] YOLOv10: Real-Time End-to-End Object Detection
[3] Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
[4] Ultralytics object-detection performance metrics

