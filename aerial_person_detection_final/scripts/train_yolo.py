"""Train a one-class YOLO model on the converted VisDrone dataset.

Examples:
  python3 scripts/train_yolo.py --model yolov8n.pt --name yolov8n_person
  python3 scripts/train_yolo.py --model yolov10n.pt --name yolov10n_person
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Base model or checkpoint, e.g. yolov8n.pt or yolov10n.pt')
    parser.add_argument('--data', default='configs/visdrone_person.yaml')
    parser.add_argument('--name', required=True)
    parser.add_argument('--project', default='results/training')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=-1, help='Use -1 for auto batch sizing')
    parser.add_argument('--device', default='0', help='CUDA device such as 0, or cpu')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f'Dataset YAML not found: {data.resolve()}')
    seed_everything(args.seed)

    model = YOLO(args.model)
    result = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer='auto',
        deterministic=True,
        amp=True,
        save=True,
        plots=True,
        resume=args.resume,
    )
    save_dir = Path(getattr(result, 'save_dir', Path(args.project) / args.name))
    best = save_dir / 'weights' / 'best.pt'
    print(f'Training finished. Best checkpoint: {best.resolve()}')


if __name__ == '__main__':
    main()
