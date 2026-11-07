"""Evaluate a frozen checkpoint on labeled simulation data."""
import argparse
import csv
from torch.utils.data import DataLoader
from .util import fresh_directory, resolve_device, sha256, write_json, IQDataset, evaluate, load_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = resolve_device(a.device)
    model, ckpt = load_model(a.checkpoint, device)
    data = IQDataset(a.manifest, a.split, ckpt["config"], ckpt["normalization"])
    metrics, rows = evaluate(model, DataLoader(data, batch_size=ckpt["config"]["batch_size"]), ckpt["normalization"], device)
    out = fresh_directory(a.out)
    metrics.update({"split": a.split, "checkpoint_sha256": sha256(a.checkpoint),
                    "manifest_sha256": sha256(a.manifest), "selected_epoch": ckpt["epoch"]})
    write_json(out / "metrics.json", metrics)
    with (out / "per_case.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(metrics)
    return metrics


if __name__ == "__main__":
    main()
