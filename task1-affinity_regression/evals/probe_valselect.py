"""probe_valselect.py — per-epoch VAL-metric model selection for a frozen-encoder run.

Answers: "is epoch E the legitimate best by the VALIDATION metric (not test-peeking)?"
For each cached feature epoch, runs the SAME probe as tta_6set_probe (MLP, mse, 5 seeds,
early-stop on v2 VAL Spearman) and records the mean best VAL Spearman alongside the test
FULL / CL3-ID30 Spearman. The v2 val split (n=817) is disjoint from the 6 test cohorts, so
argmax(val-rho) is a valid, test-independent checkpoint selection.

CPU-only (features are cached); throttle with OMP_NUM_THREADS=2 nice -19.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -19 python -u evals/probe_valselect.py \
    --name 260915_multimae_k025_da05 --epochs 5,10,15,20,25,30,40,49 --cond maskdens
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

torch.set_num_threads(2)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tta_6set_probe import load_feats, load_pK, v2_split, cohorts, metrics, MLP, HP  # noqa: E402

DEVICE = "cpu"
FD = "dataset/data/pdbbind/features"


def train_predict_val(feats, pK, v2, test_pids, seed):
    """Same protocol as tta_6set_probe.train_predict, but ALSO return best val-Spearman."""
    torch.manual_seed(seed); np.random.seed(seed)
    loss_fn = nn.MSELoss()

    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        X = np.stack([feats[p] for p in pids]).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        return X, y, pids

    tr = [p for p, s in v2.items() if s == "train"]
    va = [p for p, s in v2.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr); Xva, yva, _ = arrs(va); Xte, yte, te_pids = arrs(test_pids)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    prep = lambda X: torch.tensor((X - mu) / sd, device=DEVICE)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)
    model = MLP(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best, best_state, bad = -1e9, None, 0
    n = Xtr_t.size(0)
    for _ in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, HP["bs"]):
            idx = perm[i:i + HP["bs"]]
            if idx.numel() < 4:
                continue
            opt.zero_grad(); loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * ys + ym
        vs = float(spearmanr(pv, yva).statistic)
        if not np.isfinite(vs):
            vs = -1.0
        if vs > best + 1e-4:
            best, best_state, bad = vs, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return dict(zip(te_pids, pte)), dict(zip(te_pids, yte)), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", default="5,10,15,20,25,30,40,49")
    ap.add_argument("--cond", default="maskdens", help="maskdens = NO-density; '' = WITH-density")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    pK, v2 = load_pK(), v2_split()
    coh = cohorts(v2)
    test_pool = set().union(*coh.values())
    suffix = f"_{args.cond}" if args.cond else ""
    eps = [int(e) for e in args.epochs.split(",")]

    print(f"run={args.name}  cond={'NO-density' if args.cond else 'WITH-density'}  seeds={args.seeds}")
    print(f"{'epoch':>5} {'VAL-rho':>16} {'test-FULL':>16} {'test-CL3ID30':>16}")
    print("-" * 56)
    rows = []
    for ep in eps:
        f = f"{FD}/atomblob_density_gradmag_e{ep}_v5_{args.name}{suffix}.pt"
        if not os.path.exists(f):
            print(f"{ep:>5}   (no cached features)"); continue
        feats = load_feats(f)
        vals, fulls, nov = [], [], []
        for s in range(args.seeds):
            pred, ytrue, bv = train_predict_val(feats, pK, v2, test_pool, s)
            vals.append(bv)
            mf = metrics(pred, ytrue, coh["FULL"]);      fulls.append(mf[1]) if mf else None
            mn = metrics(pred, ytrue, coh["CL3-ID30"]);  nov.append(mn[1]) if mn else None
        rows.append((ep, np.mean(vals), np.std(vals), np.mean(fulls), np.std(fulls), np.mean(nov)))
        print(f"{ep:>5} {np.mean(vals):>8.4f}±{np.std(vals):.3f} "
              f"{np.mean(fulls):>8.4f}±{np.std(fulls):.3f} {np.mean(nov):>10.4f}", flush=True)

    if rows:
        bv = max(rows, key=lambda r: r[1])
        print("-" * 56)
        print(f"ARGMAX VAL-rho = e{bv[0]}  (val {bv[1]:.4f} -> test-FULL {bv[3]:.4f}, CL3-ID30 {bv[5]:.4f})")
        for r in rows:
            if r[0] == 25:
                print(f"e25            val {r[1]:.4f} -> test-FULL {r[3]:.4f}, CL3-ID30 {r[5]:.4f}"
                      f"   {'(== argmax, e25 IS val-best)' if bv[0]==25 else '(NOT argmax)'}")


if __name__ == "__main__":
    main()
