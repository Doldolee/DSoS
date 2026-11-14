"""Train a fresh mixed-simulation U-Net; choose checkpoints using validation only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .util import (fresh_directory, resolve_device, sha256, write_json, IQDataset,
                   fit_normalization, evaluate, validate_joint_config, validate_checkpoint,
                   CHECKPOINT_FORMAT_VERSION)
from .model import UNet


def save_checkpoint(path, value):
    temp = path.with_name(path.name + ".partial")
    torch.save(value, temp)
    temp.replace(path)


def plot_history(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (out/"history.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
    epochs = [r["epoch"] for r in records]
    axes[0].plot(epochs, [r["train_normalized_l1"] for r in records])
    axes[0].set(xlabel="Epoch", ylabel="Normalized L1", title="Training loss")
    axes[1].plot(epochs, [r["validation"]["macro_sos_mae_m_s"] for r in records])
    axes[1].set(xlabel="Epoch", ylabel="MAE (m/s)", title="SoS validation: equal-domain mean")
    axes[2].plot(epochs, [r["validation"]["macro_att_mae_db_cm"] for r in records])
    axes[2].set(xlabel="Epoch", ylabel="MAE (dB/cm at 5 MHz)", title="ATT validation: equal-domain mean")
    fig.savefig(out/"history.png", dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", default="configs/unet.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", help="Trusted last.pt from this run")
    a = p.parse_args(argv)
    config = json.loads(Path(a.config).read_text())
    validate_joint_config(config)
    if a.epochs is not None:
        config["epochs"] = a.epochs
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = resolve_device(a.device)
    digest = sha256(a.manifest)
    out = Path(a.out)
    previous = None
    if a.resume:
        previous = torch.load(a.resume, map_location="cpu", weights_only=True)
        validate_checkpoint(previous)
        if previous["manifest_sha256"] != digest:
            raise ValueError("Resume data manifest differs")
        if any(config[k] != previous["config"][k] for k in config if k != "epochs"):
            raise ValueError("Resume configuration differs (only epochs may change)")
        if not out.is_dir() or Path(a.resume).resolve().parent != out.resolve():
            raise ValueError("Resume must use the original run directory")
        norm = previous["normalization"]
    else:
        fresh_directory(out)
        norm = fit_normalization(a.manifest)
    train = IQDataset(a.manifest, "train", config, norm)
    val = IQDataset(a.manifest, "val", config, norm)
    if {r["dataset"] for r in train.rows} != {"feigin", "quasos"}:
        raise ValueError("The student protocol trains on BOTH feigin and quasos")
    if {r["dataset"] for r in val.rows} != {"feigin", "quasos"}:
        raise ValueError("Validation must include BOTH feigin and quasos")
    counts = Counter(r["dataset"] for r in train.rows)
    model = UNet(config["base_channels"], config["output_shape"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    start, best = 0, float("inf")
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start, best = previous["epoch"], previous["best_val_normalized_mae"]
    if config["epochs"] <= start:
        raise ValueError("Requested epochs must exceed the last completed epoch")
    write_json(out / "config.json", config)
    write_json(out / "normalization.json", norm)
    write_json(out / "data_receipt.json", {"manifest_sha256": digest, "training_cases": dict(counts),
              "loss": "equal-weight mean L1 of separately standardized SoS and ATT",
              "selection": "minimum equal-domain, equal-target validation MAE normalized by training target SD; no test selection",
              "balanced_domain_sampling": config["balanced_domains"], "device": str(device),
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)})
    val_loader = DataLoader(val, batch_size=config["batch_size"], num_workers=config["num_workers"])
    for epoch in range(start + 1, config["epochs"] + 1):
        g = torch.Generator().manual_seed(seed + epoch)
        sampler = None
        if config["balanced_domains"]:
            sampler = WeightedRandomSampler([1/counts[r["dataset"]] for r in train.rows], len(train), replacement=True, generator=g)
        loader = DataLoader(train, batch_size=config["batch_size"], sampler=sampler,
                            shuffle=sampler is None, generator=g, num_workers=config["num_workers"])
        model.train()
        total = n = 0
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            total += loss.item() * len(x); n += len(x)
        metrics, _ = evaluate(model, val_loader, norm, device)
        improved = metrics["macro_normalized_mae"] < best
        best = min(best, metrics["macro_normalized_mae"])
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                      "best_val_normalized_mae": best, "config": config, "normalization": norm,
                      "manifest_sha256": digest, "validation": metrics, "format_version": CHECKPOINT_FORMAT_VERSION}
        if improved:
            save_checkpoint(out / "best.pt", checkpoint)
        save_checkpoint(out / "last.pt", checkpoint)
        record = {"epoch": epoch, "train_normalized_l1": total/n, "validation": metrics, "best": improved}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch {epoch}/{config['epochs']}: train L1={total/n:.5f}, "
              f"val SoS MAE={metrics['macro_sos_mae_m_s']:.3f} m/s, "
              f"ATT MAE={metrics['macro_att_mae_db_cm']:.4f} dB/cm at 5 MHz, "
              f"normalized score={metrics['macro_normalized_mae']:.5f}", flush=True)
    plot_history(out)
    write_json(out / "training_complete.json", {"state": "complete", "epochs": config["epochs"],
               "best_checkpoint_sha256": sha256(out / "best.pt"), "test_evaluated": False,
               "scope": "training and validation complete; run evaluate separately"})
    return out


if __name__ == "__main__":
    main()
