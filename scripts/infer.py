"""Save a physical-unit SoS map from an IQ NPZ, with optional reference preview."""
import argparse
import numpy as np
import torch
from .util import fresh_directory, resolve_device, sha256, write_json, input_tensor, load_model


def predict(model, sample, ckpt, device):
    x = input_tensor(sample["iq"])
    if list(x.shape) != ckpt["config"]["input_shape"]:
        raise ValueError("Input IQ shape differs from the training configuration")
    with torch.no_grad():
        y = model(x[None].to(device))[0, 0].cpu().numpy()
    n = ckpt["normalization"]
    y = y * n["sos_std_m_s"] + n["sos_mean_m_s"]
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite prediction")
    return y


def plot_prediction(path, sample, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    has_gt = "sos" in sample
    fig, axes = plt.subplots(1, 3 if has_gt else 1, figsize=(11 if has_gt else 4, 5), squeeze=False, constrained_layout=True)
    x, z = np.asarray(sample["x_m"])*1000, np.asarray(sample["z_m"])*1000
    extent = [x[0], x[-1], z[-1], z[0]]
    panels = [("Predicted SoS", prediction, "viridis", 1300, 1800)]
    if has_gt:
        panels = [("Simulation GT", sample["sos"], "viridis", 1300, 1800)] + panels + [("Absolute error", abs(prediction-sample["sos"]), "magma", 0, None)]
    for ax, (title, data, cmap, vmin, vmax) in zip(axes[0], panels):
        im = ax.imshow(data, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
        fig.colorbar(im, ax=ax, label="m/s")
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
    np.save(out / "sos_prediction_m_s.npy", y, allow_pickle=False)
    plot_prediction(out / "prediction.png", sample, y)
    write_json(out / "prediction.json", {"units": "m/s", "axis_order": ["z", "x"],
               "checkpoint_sha256": sha256(a.checkpoint), "input_sha256": sha256(a.input),
               "shape": list(y.shape), "reference_available": "sos" in sample})
    return out


if __name__ == "__main__":
    main()
