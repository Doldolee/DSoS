"""Save physical-unit SoS and ATT maps from an IQ NPZ, with optional references."""
import argparse
import numpy as np
import torch
from .util import (fresh_directory, resolve_device, sha256, write_json, input_tensor,
                   load_model, denormalize_targets, TARGETS, TARGET_UNITS)


def predict(model, sample, ckpt, device):
    x = input_tensor(sample["iq"])
    if list(x.shape) != ckpt["config"]["input_shape"]:
        raise ValueError("Input IQ shape differs from the training configuration")
    with torch.no_grad():
        y = model(x[None].to(device))[0].cpu().numpy()
    if list(y.shape) != [len(TARGETS), *ckpt["config"]["output_shape"]]:
        raise ValueError("Model output shape differs from the joint training configuration")
    y = denormalize_targets(y, ckpt["normalization"])
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite prediction")
    return {key: y[j] for j, key in enumerate(TARGETS)}


def plot_prediction(path, sample, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_gt = any(key in sample for key in TARGETS)
    fig, axes = plt.subplots(2, 3 if has_gt else 1, figsize=(11 if has_gt else 4, 8), squeeze=False, constrained_layout=True)
    x, z = np.asarray(sample["x_m"])*1000, np.asarray(sample["z_m"])*1000
    extent = [x[0], x[-1], z[-1], z[0]]
    for j, key in enumerate(TARGETS):
        title = "SoS" if key == "sos" else "ATT"
        bounds = (1300, 1800) if key == "sos" else (0, max(1.0, float(np.max(prediction[key])), float(np.max(sample[key])) if key in sample else 0))
        panels = [(f"Predicted {title}", prediction[key], "viridis", *bounds)]
        if has_gt:
            panels = [(f"{title} reference", sample.get(key), "viridis", *bounds)] + panels
            panels += [("Absolute error", abs(prediction[key]-sample[key]) if key in sample else None, "magma", 0, None)]
        for ax, (label, data, cmap, vmin, vmax) in zip(axes[j], panels):
            if data is None:
                ax.set_axis_off()
                ax.set_title(f"{title}: no reference")
                continue
            im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set(title=label, xlabel="x (mm)", ylabel="z (mm)")
            fig.colorbar(im, ax=ax, label=TARGET_UNITS[key])
    fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="NPZ containing iq, x_m, z_m")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    device = resolve_device(a.device)
    model, ckpt = load_model(a.checkpoint, device)
    with np.load(a.input, allow_pickle=False) as f:
        sample = {k: f[k] for k in f.files}
    y = predict(model, sample, ckpt, device)
    out = fresh_directory(a.out)
    np.save(out / "sos_prediction_m_s.npy", y["sos"], allow_pickle=False)
    np.save(out / "att_prediction_db_cm.npy", y["att"], allow_pickle=False)
    plot_prediction(out / "prediction.png", sample, y)
    write_json(out / "prediction.json", {"units": TARGET_UNITS, "target_order": list(TARGETS), "axis_order": ["z", "x"],
               "checkpoint_sha256": sha256(a.checkpoint), "input_sha256": sha256(a.input),
               "shape_per_target": list(y["sos"].shape),
               "files": {"sos": "sos_prediction_m_s.npy", "att": "att_prediction_db_cm.npy"},
               "reference_available": {key: key in sample for key in TARGETS}})
    return out


if __name__ == "__main__":
    main()
