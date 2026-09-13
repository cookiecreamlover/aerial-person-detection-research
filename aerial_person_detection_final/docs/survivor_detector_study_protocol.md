# A Reproducible Comparison of YOLOv8, YOLOv10, and Faster R-CNN for Aerial Human Detection in a Drone-Based Survivor Search System

## Study decision

This project will use the **survivor-search system** as the application context. The primary research task is **RGB aerial human detection**, treated as a proxy for survivor detection. A detected person is a rescue candidate, not a confirmed survivor. The study evaluates feasibility under controlled image-domain and computational constraints; it does not claim deployment on a physical UAV. The wildlife system and thermal stream are excluded from the primary model comparison.

The principal dataset will be **VisDrone2019-DET**. Its images are captured from drones and contain dense, small, and occluded targets across varied locations, lighting conditions, weather conditions, and scene densities [1]. The study will merge the VisDrone `pedestrian` and `people` classes into a single `person` class. This avoids pretending that posture is a meaningful distinction for initial survivor search.

## Research question

Under the same training data, image resolution, confidence-selection rule, hardware, and measurement procedure, which of YOLOv8n, YOLOv10n, and Faster R-CNN with ResNet-50-FPN provides the best operational trade-off between detection quality and deployment cost for aerial person detection?

The contribution is a **deployment-oriented experimental framework for selecting aerial person detectors under simultaneous accuracy and computational constraints**. It is not a new detector architecture.

## Evidence layers and claim boundaries

| Evidence layer | Data or procedure | Permitted claim |
|---|---|---|
| Aerial detection | Real labelled VisDrone images | Detector quality on aerial imagery |
| Hardware feasibility | Benchmark on a declared target computer | Computational feasibility for possible embedded use |
| UAV pipeline | Recorded aerial frames processed through a simulated flight path | Pipeline integration demonstration only |

The paper must not claim that a model achieved a particular frame rate on a drone, that the system was field-validated, or that autonomous flight is safe. A physical UAV experiment is outside this study.

## Hypotheses

H1. YOLOv8n and YOLOv10n will provide higher throughput and lower latency than Faster R-CNN.

H2. Faster R-CNN may provide competitive localization quality but will have greater latency and memory cost.

H3. The operationally preferred detector will depend on the mission constraint rather than on one aggregate accuracy metric. A real-time search aircraft and an offline mapping workflow may select different models.

## Models

| Model | Role in comparison | Required control |
|---|---|---|
| YOLOv8n | Lightweight one-stage baseline | Same one-class fine-tuning data and image size as other models |
| YOLOv10n | NMS-free real-time detector | Same one-class fine-tuning data and image size as other models |
| Faster R-CNN ResNet-50-FPN | Two-stage accuracy-oriented baseline | Same data split, image normalization policy, and evaluation images |

The models must be fine-tuned on the same converted one-class dataset. COCO-pretrained weights may be used only as initialization. They must not be compared as unmodified COCO classifiers against a drone-specific test set.

## Dataset protocol

Use the fixed VisDrone2019-DET train, validation, and test-dev splits. Convert every annotation whose original class is `pedestrian` or `people` into `person`. Ignore the original `others` and ignored-region annotations unless a separate sensitivity experiment is declared.

The primary split is:

| Split | Purpose |
|---|---|
| Train | Model fitting |
| Validation | Early stopping and confidence-threshold selection |
| Test-dev | One final locked evaluation after all decisions are frozen |

The test-dev set must not be used for model selection, threshold tuning, augmentation selection, or repeated debugging. The VisDrone challenge test set is not used because its ground-truth annotations are unavailable.

Before training, verify that the local test-dev archive includes the ground-truth annotations required by the evaluator. If it does not, freeze a sequence-aware held-out split from the annotated material and update the configuration before any model is trained. The validation split must not become the test split after threshold selection.

## Controlled variables

The following values must be identical across the three models wherever the framework permits:

| Variable | Frozen value |
|---|---|
| Target classes | One class: `person` |
| Dataset | VisDrone2019-DET converted to one class |
| Test images | Exactly the same test-dev image list |
| Input size | 1280 × 1280 for the primary experiment; 640 × 640 sensitivity experiment |
| Confidence rule | Generate raw detections; select one threshold on validation data using the F1-maximizing threshold; lock it before test evaluation |
| IoU matching threshold | 0.50 for P/R/F1; report COCO-style AP across 0.50:0.05:0.95 |
| Hardware | One declared machine and one declared device |
| Software | Pinned Python, PyTorch, Torchvision, Ultralytics, CUDA, and driver versions |
| Timing | Warm-up excluded; at least 100 timed test images; report median, p95, and mean |
| Batch size | 1 for deployment latency; a separate throughput experiment may vary batch size |
| Augmentation policy | Same semantic augmentation policy; framework-specific syntax must be documented |
| Randomness | Fixed seeds; repeat fine-tuning with at least three seeds when compute permits |

A framework may require different tensor normalization or internal preprocessing. Such differences must be reported, but the external image resolution, image list, and batch size remain fixed.

## Primary accuracy metrics

Generate and preserve all raw model detections on the locked test-dev set. Report the following on that set:

1. Precision, recall, and F1 at the validation-selected threshold.
2. AP50.
3. AP50–95 using IoU thresholds from 0.50 through 0.95 in increments of 0.05.
4. Per-size performance for small, medium, and large targets where the evaluation implementation supports it.
5. False positives per image and missed-person rate.
6. Precision–recall curves and F1-versus-threshold curves.

Precision, recall, and F1 are operating-point metrics computed after applying the validation-selected threshold. AP50 and mAP50–95 must be computed from all raw detections ranked by confidence, without applying the operating threshold first. AP summarizes the precision–recall curve, while mAP50–95 averages AP over multiple IoU thresholds [4].

## Efficiency metrics

Measure efficiency on the same declared target computer after model loading. The preferred embedded target is a Raspberry Pi 4 8GB, if available; otherwise declare the exact substitute hardware. Record the CPU/GPU model, RAM, CUDA and driver versions, Python, PyTorch, Torchvision, and Ultralytics versions, input size, batch size, warm-up count, number of timed repetitions, and whether model loading, disk I/O, and visualization are excluded. Use at least 20 warm-up frames and at least 100 timed samples. GPU synchronization must be performed before and after each timed call when CUDA is used. A Pi result supports a claim about that computer, not about a drone.

| Metric | Definition |
|---|---|
| Parameter count | Number of trainable and total parameters reported separately when available |
| Weight size | Serialized checkpoint size in MB |
| Peak memory | Peak GPU memory during inference; peak RSS on CPU runs |
| End-to-end latency | Framework model-call time including the framework's preprocessing, inference, and output postprocessing |
| Timing variation | Mean, standard deviation, median, and p95 over repeated image-level measurements |
| Throughput | Images per second at batch size 1 and the declared deployment batch size |
| Energy, optional | Joules per image if a reproducible power meter is available |

The primary implementation reports end-to-end model-call timing rather than pretending that framework-specific preprocessing, forward-pass, and postprocessing stages are directly comparable. If component-level timing is added later, it must use framework-specific instrumentation and be reported as a separate analysis. For GPU timing, synchronize the device before starting and after completing each measurement. Report the hardware, precision mode, driver, and whether preprocessing runs on CPU or GPU. Do not compare one model using GPU timing with another using CPU timing.

## Operational metrics

The paper should report two deployment views rather than one arbitrary weighted score.

### Real-time rescue view

Select the best model subject to:

- end-to-end latency ≤ 100 ms per image;
- recall ≥ 0.80 on the locked test set;
- the smallest model size among models satisfying both constraints.

If no model satisfies the constraints, report that result directly.

### Accuracy-first mapping view

Select the model with the highest AP50–95. Report its latency and memory cost beside the accuracy value. Do not call it universally best.

A Pareto plot of AP50–95 versus latency should identify models that are not dominated by another model. A second Pareto plot should use recall versus end-to-end latency.

## Statistical reporting

For each saved checkpoint, the scoring script performs image-level bootstrap resampling and writes 95% confidence intervals for precision, recall, F1, AP50, and mAP50–95. These intervals estimate test-image sampling uncertainty. They are not a substitute for training-seed variation. If compute permits, train each model with at least three seeds and report seed-level mean and confidence intervals as a separate analysis.

Use paired comparisons because every model sees the same test images. Report the bootstrap distribution of metric differences, for example YOLOv10n minus YOLOv8n AP50–95. Avoid claiming a meaningful difference from overlapping point estimates alone.

## Ablations

The primary paper should include only a small number of interpretable ablations:

1. Input size: 640 versus 1280.
2. Deployment precision: FP32 versus FP16 when hardware supports it.
3. Confidence threshold: fixed default versus validation-selected threshold.
4. Optional thermal engineering extension: RGB-only versus a separately labelled thermal or fused dataset. This extension must never be mixed into the RGB primary result.

## Required visual analysis

The paper should include four figures. The first should show the survivor-search architecture from RGB frame to candidate-person alert. The second should show the same difficult VisDrone images with ground truth, YOLOv8n, YOLOv10n, and Faster R-CNN predictions in aligned panels. The third should plot AP50–95 against end-to-end latency, with point size representing checkpoint size. The fourth should show failure examples grouped by tiny target size, occlusion, dense scenes, low contrast, and background confusion. These figures should support an explanation of why the models differ, rather than only displaying a ranking.

## Simulated UAV systems demonstration

After the locked detector benchmark, a recorded aerial-image sequence may be passed through a simulated pipeline of RGB frame, detector output, candidate filtering, simple temporal candidate association, candidate ID, optional coordinate passthrough, and alert-ready record. This demonstrates integration with a UAV search concept but is not detector ground truth, physical flight testing, field validation, or a safety assessment. The association method is not a robust multi-object tracker and may fail under crossings, camera motion, temporary disappearance, or scale change. Its output must be labelled `simulated_uav_pipeline_demonstration`.

## Metric implementation validation

Before reporting results, run the optional `evaluation/crosscheck_cocoeval.py` utility with `pycocotools` installed on the same raw predictions and annotations. Compare custom AP50 and mAP50–95 with COCOeval, investigate any material discrepancy, and preserve the cross-check output with the experiment artifacts.

## Repository reproducibility check

Run `python3 scripts/run_checks.py` from the repository root before releasing the code. The study regression tests cover class mapping, AP calculation, bootstrap confidence intervals, and evidence-label preservation. Passing these tests does not substitute for the real VisDrone experiment.

## What the current repository can and cannot claim

The repository currently contains simulation tests, detector adapters, metric implementations, and a synthetic stress benchmark. The synthetic benchmark is not model accuracy and must not appear as the primary result table. Real results can be generated only after obtaining the VisDrone images and annotations, converting them with the supplied preparation script, fine-tuning all three models, and running the locked benchmark.

The survivor simulation remains useful as a systems demonstration. It can show that detections can be fused, projected to approximate GPS coordinates, deduplicated, and delivered to a base station. It cannot establish detector accuracy or prove safe autonomous flight.

## Proposed paper contribution

The publishable contribution should be framed as a **controlled accuracy–efficiency comparison for aerial person detection in a survivor-search deployment context**, not as a new detector. The novelty is the consistent operational evaluation and deployment analysis, provided that the real experiments are completed and the limitations are reported honestly.

## References

[1]: https://github.com/VisDrone/VisDrone-Dataset "VisDrone dataset repository and benchmark description"

[2]: https://arxiv.org/abs/2405.14458 "YOLOv10: Real-Time End-to-End Object Detection"

[3]: https://arxiv.org/abs/1506.01497 "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

[4]: https://docs.ultralytics.com/guides/yolo-performance-metrics "Ultralytics object-detection performance metrics"
