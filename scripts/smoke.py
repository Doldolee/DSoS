"""CPU plumbing check with analytic TOY arrays, not medical/simulation evidence."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .util import fresh_directory, save_sample, write_json, write_manifest


def make_toy(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(17)
    for domain in ["feigin", "quasos"]:
        for split, count in [("train", 4), ("val", 1), ("test", 1)]:
            for i in range(count):
                name = f"toy_{domain}_{split}_{i}"
                iq = rng.normal(size=(2, 128, 16)).astype(np.float32)
                sos = (1540 + 20*iq[0].reshape(32, 4, 16).mean(1)).astype(np.float32)
                relative = f"cases/{name}.npz"
                save_sample(root/relative, iq=iq, rf=iq[0], sos=sos, att=np.ones_like(sos),
                            x_m=np.linspace(-.009,.009,16), z_m=np.linspace(0,.037,32))
                rows.append({"id": name, "dataset": domain, "split": split, "group": name, "path": relative})
    write_manifest(root/"manifest.jsonl", rows)
    return root/"manifest.jsonl"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True)
    a=p.parse_args(argv)
    torch.set_num_threads(2)
    out=fresh_directory(a.out)
    manifest=make_toy(out/"toy_data")
    config={"input_shape":[2,128,16],"output_shape":[32,16],"base_channels":4,"epochs":2,
            "batch_size":2,"learning_rate":.001,"weight_decay":.0001,"seed":17,"num_workers":0,"balanced_domains":True}
    write_json(out/"smoke_config.json",config)
    from .train import main as train
    from .evaluate import main as evaluate
    from .infer import main as infer
    train(["--manifest",str(manifest),"--config",str(out/"smoke_config.json"),"--out",str(out/"run"),"--device","cpu"])
    evaluate(["--manifest",str(manifest),"--checkpoint",str(out/"run/best.pt"),"--out",str(out/"test"),"--device","cpu"])
    infer(["--input",str(out/"toy_data/cases/toy_feigin_test_0.npz"),"--checkpoint",str(out/"run/best.pt"),"--out",str(out/"prediction"),"--device","cpu"])
    write_json(out/"smoke_receipt.json",{"state":"passed","scope":"analytic toy arrays: train, validation selection, checkpoint reload, test metrics, inference NPY/PNG",
               "research_performance_evidence":False})
    print(f"Smoke passed: {out}. These are TOY arrays, not ultrasound performance results.")


if __name__=="__main__":main()
