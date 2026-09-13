"""Plot AP50-95 versus latency from merged experiment results.

Expected JSON format:
{
  "models": [
    {"model": "YOLOv8", "map50_95": 0.42,
     "mean_end_to_end_latency_ms": 35.0, "checkpoint_size_mb": 12.0}
  ]
}
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="pareto_ap_latency.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot figures") from exc

    data = json.loads(Path(args.input).read_text())
    rows = data.get("models", data if isinstance(data, list) else [])
    if not rows:
        raise SystemExit("No model rows found")
    x = [r["mean_end_to_end_latency_ms"] for r in rows]
    y = [r["map50_95"] for r in rows]
    sizes = [max(40, 30 * (r.get("checkpoint_size_mb") or 1)) for r in rows]
    labels = [r["model"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=sizes, alpha=0.85)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Mean end-to-end latency (ms, lower is better)")
    plt.ylabel("mAP50–95 (higher is better)")
    plt.title("Aerial person detection accuracy–latency trade-off")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
