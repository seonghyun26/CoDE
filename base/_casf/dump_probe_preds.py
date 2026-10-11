"""Dump per-PID predictions for the VoxBind frozen-encoder probes.

The default invocation preserves the original v2 behavior.  ``--split`` also allows the same
probe recipe to be run on a cleaned split such as ``lp_edrscc_v2_cl123`` so its test predictions
can be re-scored on protein-novelty subsets without training the head again.

  CUDA_VISIBLE_DEVICES=<g> python base/_casf/dump_probe_preds.py
  python base/_casf/dump_probe_preds.py --split lp_edrscc_v2_cl123 --method C CDG
"""
import argparse, csv, importlib.util, os, sys
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("p01c", f"{REPO}/task1-affinity_regression/probe.py")
pr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pr)

OUT = f"{REPO}/base/_casf/novel_preds"; os.makedirs(OUT, exist_ok=True)
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)
CDG = "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt"  # canonical Table-1 C+D+G (probe_casf CDG_100m_mask075)
C   = "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt"
METHODS = {
    "C":        ("C",          C,   {}),
    "CDG":      ("C+D+G",      CDG, {}),
    "CDG_corr": ("C+D+G +corr", CDG, dict(probe_loss="mse+corr", aux_weight=5.0)),
}

def safe(m):  # filesystem-safe method key, shared with score_novel_subsets.py
    return "".join(ch if ch.isalnum() else "_" for ch in m)

def feats(fn):
    return torch.load(pr.FEAT_DIR / fn, weights_only=False)["features"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="lp_edrscc_v2")
    parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()

    sys.path.insert(0, REPO)
    from voxbind.splits import load_split
    split = load_split(args.split)
    split_map = {p: role for role in ("train", "val", "test") for p in split[role]}
    split_tag = "v2" if args.split == "lp_edrscc_v2" else args.split.removeprefix("lp_edrscc_v2_")
    os.makedirs(args.out_dir, exist_ok=True)

    lp = pr.load_lp_index(pr.LP_CSV)
    for method_key in args.method:
        m, fn, extra = METHODS[method_key]
        fd = feats(fn)
        data = pr.build_dataset(fd, lp, drop_covalent=True, cl1_only=False, split_map=split_map)
        pids = data["test"]["pid"]
        for s in args.seeds:
            r = pr.train_one(data, seed=s, device=args.device, **{**HP, **extra})
            out = f"{args.out_dir}/{safe(m)}_{split_tag}_seed{s}.csv"
            with open(out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["pid", "pred", "y"])
                for p, pd_, y_ in zip(pids, r["_pred_te"], r["_yte"]):
                    w.writerow([p, float(pd_), float(y_)])
            print(f"{m:14} seed{s}: rho={r['test_spearman']:.3f} r={r['test_pearson']:.3f} "
                  f"rmse={r['test_rmse']:.3f} n={len(pids)}")

if __name__ == "__main__":
    main()
