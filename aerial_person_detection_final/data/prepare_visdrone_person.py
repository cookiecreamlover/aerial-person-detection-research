"""Prepare VisDrone-DET for one-class aerial person detection.

The VisDrone annotations use class 1 for pedestrian and class 2 for people.
Both are mapped to one `person` class for the survivor-search study. The
script writes YOLO labels and a JSONL evaluation manifest containing all
images in the requested split, including images with no person boxes.
"""
import argparse
import json
from pathlib import Path

PERSON_CLASSES = {1, 2}
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test-dev": "VisDrone2019-DET-test-dev"}


def convert_split(root: Path, split: str, output_root: Path, manifest_path: Path):
    source = root / SPLITS[split]
    image_dir = source / "images"
    annotation_dir = source / "annotations"
    if not image_dir.exists() or not annotation_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {annotation_dir}")

    yolo_image_dir = output_root / "images" / split
    yolo_label_dir = output_root / "labels" / split
    yolo_image_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        truths = []
        yolo_lines = []
        width = height = None
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
        except ImportError as exc:
            raise SystemExit("Pillow is required for VisDrone preparation") from exc

        if annotation_path.exists():
            for line in annotation_path.read_text().splitlines():
                fields = line.split(",")
                if len(fields) < 8:
                    continue
                x, y, w, h, _, class_id, _, _ = map(int, fields[:8])
                if class_id not in PERSON_CLASSES or w <= 0 or h <= 0:
                    continue
                cx, cy = (x + w / 2) / width, (y + h / 2) / height
                nw, nh = w / width, h / height
                yolo_lines.append(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
                truths.append({"class_id": 0, "label": "person", "x1": x, "y1": y, "x2": x + w, "y2": y + h})

        label_path = yolo_label_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        rows.append({
            "image_id": image_path.stem,
            "image": str(image_path.resolve()),
            "scenario": "survivor",
            "modality": "rgb",
            "truths": truths,
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"Wrote {len(rows)} {split} images and manifest {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="VisDrone2019-DET root containing split directories")
    parser.add_argument("--split", choices=sorted(SPLITS), default="test-dev")
    parser.add_argument("--output-root", default="data/visdrone_person")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    output_root = Path(args.output_root)
    manifest = Path(args.manifest or output_root / f"{args.split}.jsonl")
    convert_split(root, args.split, output_root, manifest)


if __name__ == "__main__":
    main()
