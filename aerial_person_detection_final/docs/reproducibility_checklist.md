# Reproducibility checklist

Before reporting the comparison, record the VisDrone dataset release, image and annotation counts, the `pedestrian + people → person` mapping, the official split names, the test-dev annotation availability, model weight files and hashes, framework versions, input resolution, confidence-threshold rule, IoU threshold, device, batch size, warm-up policy, timing repetitions, CUDA synchronization policy, and the end-to-end timing scope.

Record the complete training command for each model, including seed, epochs, optimizer, learning-rate schedule, augmentation settings, batch size, image size, and checkpoint path. If multiple seeds are trained, preserve every seed-specific checkpoint and result file.

For deployment-feasibility claims, save `results/environment.json` from `scripts/capture_environment.py`. Report the exact CPU/GPU, RAM, operating system, Python, PyTorch, Torchvision, Ultralytics, CUDA, and driver versions. A Raspberry Pi result supports a claim about that Pi configuration, not about a physical drone.

Every result must be reproducible from a command, a manifest, and a saved configuration. Predictions from the real RGB benchmark must retain the label `real_rgb_evaluation`. Synthetic results from earlier prototypes must retain `synthetic_stress_test`. Video systems demonstrations must use `real_rgb_video_sequence`, and simulated pipeline output must use `simulated_uav_pipeline_demonstration`.

For each saved test prediction file, run the scoring script with a declared bootstrap iteration count and seed. Preserve the resulting image-bootstrap confidence intervals and distinguish them from training-seed variation. Preserve the raw per-image predictions so that qualitative examples and failure analyses can be regenerated.

Before publication, run the optional COCOeval cross-check on identical annotations and raw predictions, preserve its output, and investigate any material disagreement with the custom metric implementation.

The primary study does not claim thermal performance, wildlife performance, GPS accuracy, robust multi-object tracking, physical UAV deployment, or autonomous-flight safety.
