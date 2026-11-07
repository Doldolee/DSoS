"""Simulation data utilities: IO, RF/IQ processing, datasets, preparation and metrics.

Command line: python -m scritps.util --help
"""
import argparse
import csv
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
from torch.utils.data import Dataset




DX_M = 3.676470441860147e-5


XI = 322 + np.floor(np.arange(128) / (256 / 1016)).astype(int)


ZI = 60 + np.floor(np.arange(256) / (256 / 1016)).astype(int)


X_M = (XI - 575.5) * DX_M


Z_M = (ZI - 59) * DX_M


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_manifest(path, rows):
    path = Path(path)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))


def fresh_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Use a new output directory: {path}")
    path.mkdir(parents=True)
    return path


def save_sample(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError(f"Object/pickle arrays are not allowed: {key}")
        if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
            raise ValueError(f"Nonfinite sample field: {key}")
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)


def resolve_device(name):
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


FS_HZ = 40e6


START = 42


N_TIME = 2048


def resample_rf(rf, source_fs):
    ratio = Fraction(float(FS_HZ / source_fs)).limit_denominator(10000)
    if abs(float(ratio) - FS_HZ / source_fs) > 1e-8:
        raise ValueError("Unsupported sample rate ratio")
    if ratio == 1:
        return np.asarray(rf, dtype=np.float32)
    return resample_poly(rf, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def rf_to_iq(rf, center_hz, start_sample=START):
    """[rx,time=2048] real RF -> [I/Q=2,time,rx], float32.

    Fixed depth gain, complex demodulation at the *recorded* carrier, zero-phase
    lowpass. Carrier time includes the crop offset. No target is consulted.
    """
    a = np.asarray(rf, dtype=np.float32)
    if a.ndim != 2 or a.shape[-1] != N_TIME or not np.isfinite(a).all():
        raise ValueError("Expected finite [rx,2048] RF")
    if not 0 < center_hz < FS_HZ / 2:
        raise ValueError("Invalid carrier frequency")
    gain = 10.0 ** np.linspace(0, 1.8, N_TIME)
    carrier = np.exp(-2j * np.pi * center_hz * (start_sample + np.arange(N_TIME)) / FS_HZ)
    mixed = 2 * a * gain * carrier
    filt = butter(5, min(5e6, center_hz), fs=FS_HZ, output="sos")
    iq = sosfiltfilt(filt, mixed, axis=-1).T
    return np.stack([iq.real, iq.imag]).astype(np.float32)


def nearest_indices(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if np.any(np.diff(source) <= 0) or target.min() < source.min() or target.max() > source.max():
        raise ValueError("Output grid outside recorded physical support")
    hi = np.clip(np.searchsorted(source, target), 1, len(source) - 1)
    lo = hi - 1
    return np.where(target - source[lo] <= source[hi] - target, lo, hi)


def read_rows(manifest):
    manifest = Path(manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seen, groups = set(), defaultdict(set)
    for r in rows:
        for key in ("id", "dataset", "split", "group", "path"):
            if not r.get(key):
                raise ValueError(f"Missing {key} in manifest row")
        key = (r["dataset"], r["id"])
        if key in seen:
            raise ValueError(f"Duplicate sample: {key}")
        seen.add(key)
        groups[(r["dataset"], r["group"])].add(r["split"])
        relative = Path(r["path"])
        resolved = (manifest.parent / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(manifest.parent):
            raise ValueError("Sample paths must remain inside the manifest directory")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
    leaking = [g for g, splits in groups.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Scene/participant split leakage: {leaking[:4]}")
    return rows


def input_tensor(iq):
    a = np.asarray(iq, dtype=np.float32)
    if a.ndim != 3 or a.shape[0] != 2 or not np.isfinite(a).all():
        raise ValueError("IQ must be finite [2,time,rx], with real I and Q channels")
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        raise ValueError("Empty/zero input signal")
    # Per-acquisition normalization uses the signal alone, never SoS labels.
    return torch.from_numpy(np.ascontiguousarray(a / rms))


class IQDataset(Dataset):
    def __init__(self, manifest, split, config, normalization):
        self.root = Path(manifest).resolve().parent
        self.rows = [r for r in read_rows(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} samples")
        self.config, self.norm = config, normalization

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with np.load(self.root / r["path"], allow_pickle=False) as a:
            x = input_tensor(a["iq"])
            y = np.asarray(a["sos"], dtype=np.float32).copy()
        if list(x.shape) != self.config["input_shape"] or list(y.shape) != self.config["output_shape"]:
            raise ValueError(f"Shape mismatch: {r['id']}: {list(x.shape)}, {list(y.shape)}")
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite SoS target")
        y = (y - self.norm["sos_mean_m_s"]) / self.norm["sos_std_m_s"]
        return x, torch.from_numpy(y[None]), r["id"], r["dataset"]


def fit_normalization(manifest):
    """Fit target mean/std on TRAIN maps only; stream one target at a time."""
    root = Path(manifest).resolve().parent
    count = 0
    total = square = 0.0
    rows = [r for r in read_rows(manifest) if r["split"] == "train"]
    for r in rows:
        with np.load(root / r["path"], allow_pickle=False) as a:
            y = np.asarray(a["sos"], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite training target")
        count += y.size
        total += float(y.sum())
        square += float(np.square(y).sum())
    if not count:
        raise ValueError("No training targets")
    mean = total / count
    std = max(float(np.sqrt(max(square / count - mean ** 2, 0))), 1.0)
    return {"sos_mean_m_s": mean, "sos_std_m_s": std, "fit_split": "train",
            "train_cases": len(rows), "input_normalization": "per-acquisition IQ RMS, signal only"}


def finish(out, rows, info):
    write_manifest(out / "manifest.jsonl", rows)
    write_json(out / "dataset_info.json", info)
    files = sorted(p for p in out.rglob("*") if p.is_file() and not p.name.startswith("._"))
    write_json(out / "files.sha256.json", {p.relative_to(out).as_posix(): sha256(p) for p in files})


def feigin(raw, output, limit=None):
    out = fresh_directory(output)
    rows = []
    specifications = json.loads((Path(__file__).parent/"feigin_sources.json").read_text())["files"]
    for split, folder in [("train", "train"), ("val", "validation"), ("test", "test")]:
        paths = sorted((Path(raw) / folder).glob("multiplane_5_0_*.mat"))
        if not paths:
            raise FileNotFoundError(f"No 5 MHz MAT files under {Path(raw)/folder}")
        expected = {Path(e["path"]).name for e in specifications if e["split"] == split}
        if limit is None and {p.name for p in paths} != expected:
            raise ValueError(f"Incomplete {split} source files; download the full 5 MHz split, or use --limit-per-file for an explicitly limited demo")
        for path in paths:
            with h5py.File(path, "r") as f:
                rfset = f["p_f5.0_o0"]
                if rfset.ndim != 4 or rfset.shape[1:3] != (1, 128) or f["c0"].shape[-2:] != (1152, 1152):
                    raise ValueError("Unsupported Feigin HDF5 axes; expected [case,1,rx,time]")
                count = min(len(rfset), limit) if limit else len(rfset)
                for i in range(count):
                    rf = np.asarray(rfset[i, 0, 32:96, START:START+N_TIME], dtype=np.float32)
                    # h5py permits one fancy index at a time; read only the small ROI.
                    sos = np.asarray(f["c0"][i, 0, 322:830, 60:1076])[XI-322][:, ZI-60].T.astype(np.float32)
                    att = (np.asarray(f["alpha_coeff"][i, 0, 322:830, 60:1076])[XI-322][:, ZI-60].T * 25).astype(np.float32)
                    name = f"{path.stem}_{i:04d}"
                    relative = f"cases/{name}.npz"
                    save_sample(out / relative, rf=rf.T, iq=rf_to_iq(rf, 5e6), sos=sos, att=att,
                                x_m=X_M, z_m=Z_M, fs_hz=FS_HZ, fc_hz=5e6,
                                rx_m=(np.arange(128)[32:96]-63.5)*8*3.676470441860147e-5,
                                t0_s=START/FS_HZ, angle_deg=0.)
                    rows.append({"id": name, "dataset": "feigin", "split": split, "group": name, "path": relative})
            print(f"converted {path.name}: {count} cases", flush=True)
    finish(out, rows, {"dataset": "feigin", "source": "laughingrice/Ultrasound_planewave_sos_inversion",
           "frequency_hz": 5e6, "plane_wave_angle_deg": 0, "native_rx_slice": [32, 96],
           "rf_shape": [2048, 64], "iq_shape": [2, 2048, 64], "target_shape_z_x": [256, 128],
           "sos_unit": "m/s", "att_unit": "dB/cm at 5 MHz", "limited_demo": limit is not None,
           "note": "ATT=alpha_coeff*25; not raw alpha_coeff. Existing simulated RF only; no new k-Wave run."})
    return out


def quasos(raw, output, limit=None):
    raw, out = Path(raw), fresh_directory(output)
    source_rows = list(csv.DictReader((raw / "manifest.csv").open()))
    rows = []
    if limit:
        source_rows = source_rows[:limit]
    for r in source_rows:
        m = json.loads((raw / r["metadata_path"]).read_text())
        fmc = np.load(raw / r["fmc_path"], mmap_mode="r", allow_pickle=False)
        if m["fmc_axis_order"] != ["rx", "tx", "t"] or fmc.shape[:2] != (128, 128) or m["t0_s"] != 0:
            raise ValueError("Unsupported native FMC geometry")
        if r["att_target_version"] != "solver_gt_v1_2026-09-08":
            raise ValueError("Corrected solver ATT label required")
        fs = 1 / float(m["dt_s"])
        fc = float(m["template"]["center_frequency_mhz"]) * 1e6
        front = int(m["kwave"].get("front_pad_samples", 0))
        need = int(np.ceil(((START+N_TIME)/FS_HZ + 5e-6)*fs))
        if front + need > fmc.shape[-1]:
            raise ValueError("Insufficient native RF duration")
        # A zero-degree plane wave: sum all TX with zero delay, mean amplitude.
        # Read only central RX and required duration; never load an entire FMC.
        pw = np.mean(fmc[32:96, :, front:front+need], axis=1, dtype=np.float64).astype(np.float32)
        rf = resample_rf(pw, fs)[:, START:START+N_TIME]
        xi = nearest_indices(np.asarray(m["x_cm"])*.01, X_M)
        zi = nearest_indices(np.asarray(m["z_cm"])*.01, Z_M)
        sos = np.load(raw/r["sos_path"], allow_pickle=False)[zi[:, None], xi[None, :]].astype(np.float32)
        slope = np.load(raw/r["att_path"], allow_pickle=False)[zi[:, None], xi[None, :]]
        ref = float(r["att_reporting_frequency_mhz"])
        att = (slope * ref * (5/ref)**float(r["solver_alpha_power"])).astype(np.float32)
        lesion = np.load(raw/r["lesion_mask_path"], allow_pickle=False)[zi[:, None], xi[None, :]].astype(bool)
        name = r["root_name"] + "__" + r["case_id"]
        relative = f"cases/{name}.npz"
        save_sample(out/relative, rf=rf.T, iq=rf_to_iq(rf, fc), sos=sos, att=att, lesion=lesion,
                    x_m=X_M, z_m=Z_M, fs_hz=FS_HZ, fc_hz=fc,
                    rx_m=np.asarray(m["template"]["element_positions_mm"])[32:96]*1e-3,
                    t0_s=START/FS_HZ, angle_deg=0.)
        rows.append({"id": name, "dataset": "quasos", "split": r["split"], "group": r["scene_family"],
                     "path": relative, "probe": r["probe"], "native_fmc_source": r["fmc_path"],
                     "native_fmc_shape": list(fmc.shape), "att_source": r["att_path"]})
        del fmc
        print(f"converted QUASOS {len(rows)}/{len(source_rows)}: {name}", flush=True)
    finish(out, rows, {"dataset": "quasos", "version": "student_pw0_rx64_v1", "cases": len(rows),
           "source_manifest_sha256": sha256(raw/"manifest.csv"), "split": {s: sum(r["split"]==s for r in rows) for s in ["train", "val", "test"]},
           "rf_description": "0-degree plane-wave RF synthesized from existing FMC, TX128 mean, central RX64; not full native FMC",
           "rf_shape": [2048, 64], "iq_shape": [2, 2048, 64], "target_shape_z_x": [256, 128],
           "sos_unit": "m/s", "att_unit": "dB/cm at 5 MHz", "limited_demo": limit is not None,
           "att_target_version": "solver_gt_v1_2026-09-08", "new_acoustic_simulation": False})
    return out


def merge(manifests, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        for r in read_rows(manifest):
            r["path"] = (manifest.parent/r["path"]).relative_to(output.parent).as_posix()
            rows.append(r)
    write_manifest(output, rows)
    read_rows(output)  # fails if any scene leaks across splits
    print(f"Combined {len(rows)} samples -> {output}")




@torch.no_grad()
def evaluate(model, loader, norm, device):
    model.eval()
    stats = defaultdict(lambda: {"n": 0, "abs": 0., "square": 0., "signed": 0., "cases": 0})
    rows = []
    for x, y, ids, domains in loader:
        pred = model(x.to(device)).cpu().numpy()
        error = (pred - y.numpy()) * norm["sos_std_m_s"]
        for i, (sample_id, domain) in enumerate(zip(ids, domains)):
            e = error[i].astype(np.float64)
            if not np.isfinite(e).all():
                raise ValueError("Nonfinite model prediction")
            s = stats[domain]
            s["n"] += e.size
            s["abs"] += float(np.abs(e).sum())
            s["square"] += float(np.square(e).sum())
            s["signed"] += float(e.sum())
            s["cases"] += 1
            rows.append({"id": sample_id, "dataset": domain, "mae_m_s": float(np.abs(e).mean()),
                         "rmse_m_s": float(np.sqrt(np.square(e).mean())), "bias_m_s": float(e.mean())})
    summary = {d: {"cases": s["cases"], "pixels": s["n"], "mae_m_s": s["abs"]/s["n"],
                    "rmse_m_s": (s["square"]/s["n"])**0.5, "bias_m_s": s["signed"]/s["n"]}
               for d, s in stats.items()}
    return {"by_dataset": summary, "macro_mae_m_s": float(np.mean([s["mae_m_s"] for s in summary.values()])),
            "units": "m/s", "evaluation": "whole-map synthetic SoS accuracy"}, rows


def verify(root):
    root = Path(root).resolve()
    checksum_path = root/"files.sha256.json"
    expected = json.loads(checksum_path.read_text()) if checksum_path.is_file() else {}
    for relative, digest in expected.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Bundle hash mismatch or invalid path: {relative}")
    rows = read_rows(root/"manifest.jsonl")
    print(f"Validated {len(rows)} samples and disjoint scene splits; checked {len(expected)} optional file hashes")
    return rows


def load_model(checkpoint, device):
    from .model import UNet
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UNet(ckpt["config"]["base_channels"], ckpt["config"]["output_shape"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def download_feigin(output, list_only=False, one_file_per_split=False):
    """Download the pinned public Feigin files and verify original SHA-256."""
    source = json.loads((Path(__file__).parent/"feigin_sources.json").read_text())
    entries = source["files"]
    if one_file_per_split:
        entries = [next(e for e in entries if e["split"] == s) for s in ["train", "val", "test"]]
    for e in entries:
        print(f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{quote(e['path'])}")
        if list_only:
            continue
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=source["repository"], filename=e["path"], repo_type="dataset",
                                   revision=source["revision"], local_dir=output))
        if path.stat().st_size != e["bytes"] or sha256(path) != e["sha256"]:
            raise ValueError("Downloaded Feigin file size or SHA-256 mismatch")
        print(f"Downloaded: {path}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["feigin", "quasos"]:
        s = sub.add_parser(name, help="Convert existing simulation RF into student NPZ data")
        s.add_argument("--raw", required=True); s.add_argument("--out", required=True)
        s.add_argument("--limit-per-file" if name == "feigin" else "--limit", type=int)
    s = sub.add_parser("download", help="Download public Feigin data")
    s.add_argument("--out", required=True)
    s.add_argument("--list-only", action="store_true")
    s.add_argument("--one-file-per-split", action="store_true")
    s = sub.add_parser("merge", help="Combine Feigin and lab simulation manifests")
    s.add_argument("--manifests", nargs="+", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("verify", help="Check the sample manifest and optional local checksums")
    s.add_argument("--data", required=True)
    a = p.parse_args(argv)
    if a.command == "feigin": return feigin(a.raw, a.out, a.limit_per_file)
    if a.command == "quasos": return quasos(a.raw, a.out, a.limit)
    if a.command == "download": return download_feigin(a.out, a.list_only, a.one_file_per_split)
    if a.command == "merge": return merge(a.manifests, a.out)
    return verify(a.data)


if __name__ == "__main__":
    main()
