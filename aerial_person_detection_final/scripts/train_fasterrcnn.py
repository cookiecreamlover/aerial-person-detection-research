"""Train one-class Faster R-CNN ResNet-50-FPN from a JSONL manifest.

The saved checkpoint is a plain state_dict compatible with models/adapters.py.
The manifest must be produced by data/prepare_visdrone_person.py and contain
image paths plus `truths` entries with x1/y1/x2/y2 fields.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class ManifestDataset(Dataset):
    def __init__(self, manifest: str, augment: bool = False):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row['image']).convert('RGB')
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0
        width, height = image.size
        boxes = []
        for truth in row.get('truths', []):
            x1 = max(0.0, min(float(truth['x1']), width - 1))
            y1 = max(0.0, min(float(truth['y1']), height - 1))
            x2 = max(x1 + 1.0, min(float(truth['x2']), width))
            y2 = max(y1 + 1.0, min(float(truth['y2']), height))
            boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.augment and random.random() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
            if len(boxes):
                flipped = boxes.clone()
                flipped[:, 0] = width - boxes[:, 2]
                flipped[:, 2] = width - boxes[:, 0]
                target['boxes'] = flipped
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes=2):
    # Start from an ImageNet-pretrained backbone, then replace the detector head.
    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/visdrone_person/train.jsonl')
    parser.add_argument('--output', default='results/training/fasterrcnn_person.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    if not Path(args.manifest).exists():
        raise SystemExit(f'Manifest not found: {Path(args.manifest).resolve()}')
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    dataset = ManifestDataset(args.manifest, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=device.type == 'cuda')
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        scheduler.step()
        print(f'epoch={epoch + 1}/{args.epochs} loss={running / max(1, len(loader)):.5f}', flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f'Training finished. Checkpoint: {output.resolve()}')


if __name__ == '__main__':
    main()
