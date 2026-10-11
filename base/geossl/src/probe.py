"""Frozen GeoSSL-DDM SchNet + 2-layer-MLP probe on a VoxBind affinity split.

Same protocol as VoxBind's own C / C+D+G probes and the ProFSA baseline: the pretrained
encoder is frozen (features precomputed by extract_features.py), and only a small 2-layer
MLP head is trained on the split's train set, model-selected on val RMSE, evaluated on test.
Reports Pearson r + Spearman rho + RMSE, averaged over 5 seeds.

Usage:
    python src/probe.py --split atom3d_lba60_edrscc_v2 --tag lba60
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(BASE))                       # base/, for datasource
from datasource import LP_CSV, SPLITS_DIR, data_dir             # noqa: E402
import model_store  # noqa: E402
FEAT_PT = os.path.join(data_dir("geossl"), "schnet_feats.pt")   # base/data/geossl
RESULTS_DIR = os.path.join(BASE, "results")

SEEDS = [0, 1, 2, 3, 4]


class MLP(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_labels():
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    return dict(zip(lp["pid"].astype(str), lp["pK"].astype(float)))


def get_xy(pids, feats, labels):
    X, y, ids = [], [], []
    for p in pids:
        if p in feats and p in labels:
            X.append(feats[p]); y.append(labels[p]); ids.append(p)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), ids


def train_one(Xtr, ytr, Xva, yva, Xte, seed, device, epochs=300, bs=128, patience=50):
    torch.manual_seed(seed); np.random.seed(seed)
    # standardize
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    ztr = (ytr - ym) / ys

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, device=dev); ztr_t = torch.tensor(ztr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev); Xte_t = torch.tensor(Xte, device=dev)

    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossfn = nn.MSELoss()

    n = Xtr_t.shape[0]
    best_va, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossfn(model(Xtr_t[idx]), ztr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pva = model(Xva_t).cpu().numpy() * ys + ym
        va_rmse = float(np.sqrt(np.mean((pva - yva) ** 2)))
        if va_rmse < best_va - 1e-5:
            best_va = va_rmse; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    head = {"mlp": {k: v.cpu() for k, v in best_state.items()},
            "x_mean": mu, "x_std": sd, "y_mean": float(ym), "y_std": float(ys)}
    return pte, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split CSV name in voxbind/splits/ (train/val source)")
    ap.add_argument("--tag", required=True, help="short output tag, e.g. lba60")
    ap.add_argument("--test_from", default=None,
                    help="external test CSV name (with pid,pK cols) — e.g. casf2016_eval. "
                         "If set, test set comes from here; --split only provides train/val.")
    ap.add_argument("--nontrain_only", action="store_true",
                    help="for external test: keep only rows with in_v2train==0 (leak control)")
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()

    feats = torch.load(FEAT_PT, weights_only=False)
    labels = load_labels()
    sp = pd.read_csv(os.path.join(SPLITS_DIR, f"{args.split}.csv"))
    sp["pid"] = sp["pid"].astype(str)
    tr = sp[sp.split == "train"]["pid"].tolist()
    va = sp[sp.split == "val"]["pid"].tolist()

    if args.test_from:
        ext = pd.read_csv(os.path.join(SPLITS_DIR, f"{args.test_from}.csv"))
        ext.columns = [c.strip() for c in ext.columns]
        ext["pid"] = ext["pid"].astype(str)
        if args.nontrain_only and "in_v2train" in ext.columns:
            ext = ext[ext["in_v2train"] == 0]
        ext_lab = dict(zip(ext["pid"], ext["pK"].astype(float))) if "pK" in ext.columns else labels
        te = ext["pid"].tolist()
        te_labels = {**labels, **ext_lab}
    else:
        te = sp[sp.split == "test"]["pid"].tolist()
        te_labels = labels

    Xtr, ytr, _ = get_xy(tr, feats, labels)
    Xva, yva, _ = get_xy(va, feats, labels)
    Xte, yte, te_ids = get_xy(te, feats, te_labels)
    print(f"[{args.tag}] train {len(Xtr)} val {len(Xva)} test {len(Xte)} "
          f"(missing feats: tr {len(tr)-len(Xtr)}, va {len(va)-len(Xva)}, te {len(te)-len(Xte)})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    per_seed = {}
    for s in SEEDS:
        pte, head = train_one(Xtr, ytr, Xva, yva, Xte, s, args.device)
        # The probe head with its feature and pK scalers; a --test_from run refits it, kept apart.
        model_store.save_state("geossl", args.split, s, head, "probe.pt", config=vars(args),
                               sub=args.tag if args.test_from else None)
        r = float(pearsonr(pte, yte)[0])
        rho = float(spearmanr(pte, yte)[0])
        rmse = float(np.sqrt(np.mean((pte - yte) ** 2)))
        per_seed[str(s)] = {"pearson": r, "spearman": rho, "rmse": rmse, "n": len(yte)}
        print(f"  seed {s}: r={r:.4f} rho={rho:.4f} rmse={rmse:.4f}")
        with open(os.path.join(RESULTS_DIR, f"preds_GEOSSL_{args.tag}_seed{s}.jsonl"), "w") as fh:
            for pid, pred, gt in zip(te_ids, pte, yte):
                fh.write(json.dumps({"id": pid, "label": float(pred), "task": "PDBBind", "gt": float(gt)}) + "\n")

    def agg(key):
        v = np.array([per_seed[str(s)][key] for s in SEEDS])
        return {"mean": float(v.mean()), "std": float(v.std())}

    out = {
        "method": "GeoSSL", "split": args.tag,
        "pearson": agg("pearson"), "spearman": agg("spearman"), "rmse": agg("rmse"),
        "seeds": SEEDS, "n_test": len(yte), "per_seed": per_seed,
        "encoder": "GeoSSL-DDM SchNet (frozen) + 2-layer MLP probe",
    }
    outp = os.path.join(RESULTS_DIR, f"results_GEOSSL_{args.tag}.json")
    with open(outp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[{args.tag}] r={out['pearson']['mean']:.4f}±{out['pearson']['std']:.4f} "
          f"rho={out['spearman']['mean']:.4f}±{out['spearman']['std']:.4f} "
          f"rmse={out['rmse']['mean']:.4f}±{out['rmse']['std']:.4f} -> {outp}")


if __name__ == "__main__":
    main()
