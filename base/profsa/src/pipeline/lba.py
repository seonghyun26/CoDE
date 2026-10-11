import logging
import os
import csv
from typing import Any, List

from .screening import ScreeningLitModule

logger = logging.getLogger(__name__)


class LBALitModule(ScreeningLitModule):
    def step(self, batch: Any):
        inputs = batch["net_input"]
        outputs = self.model(**inputs)

        outputs["target"] = batch["target"]["finetune_target"]
        outputs = self.criterion(outputs)

        return outputs

    def test_step(self, batch: Any, batch_idx: int):
        # single forward; optionally capture per-pid (pid, pred, y) for external re-scoring
        inputs = batch["net_input"]
        # optional: capture the classification-head input (the [mol;pocket] complex rep)
        # for the 260820 complex-representation analysis. Gated by PROFSA_REPR_DIR.
        if os.environ.get("PROFSA_REPR_DIR") and not hasattr(self, "_repr_hook"):
            self._repr_rows = {}
            def _pre_hook(module, args_):
                self._repr_cat = args_[0].detach().float().cpu()
            self._repr_hook = self.model.classification_head.register_forward_pre_hook(_pre_hook)
        outputs = self.model(**inputs)
        if os.environ.get("PROFSA_REPR_DIR"):
            cat = getattr(self, "_repr_cat", None)
            if cat is not None:
                names = [str(n) for n in batch["pocket_name"]]
                for i, n in enumerate(names):
                    self._repr_rows[n] = cat[i].numpy()
        if os.environ.get("PROFSA_DUMP_DIR"):
            logit = outputs["logit"].detach().float().cpu().view(-1).tolist()
            tgt = batch["target"]["finetune_target"].detach().float().cpu().view(-1).tolist()
            names = [str(n) for n in batch["pocket_name"]]
            if not hasattr(self, "_dump_rows"):
                self._dump_rows = []
            self._dump_rows += list(zip(names, logit, tgt))
        outputs["target"] = batch["target"]["finetune_target"]
        outputs = self.criterion(outputs)
        return outputs

    def on_test_epoch_end(self):
        repr_dir = os.environ.get("PROFSA_REPR_DIR")
        if repr_dir and getattr(self, "_repr_rows", None):
            import torch
            seed = os.environ.get("PROFSA_DUMP_SEED", "0")
            os.makedirs(repr_dir, exist_ok=True)
            feats = {k: v for k, v in self._repr_rows.items()}
            dim = int(next(iter(feats.values())).shape[0])
            out = os.path.join(repr_dir, f"repr_lp_edrscc_v2_test_seed{seed}.pt")
            torch.save({"features": feats, "dim": dim, "seed": seed,
                        "split": "test", "model": "ProFSA"}, out)
            logger.info(f"[repr] wrote {len(feats)} x {dim}-d complex reps -> {out}")
            self._repr_rows = {}

        dump = os.environ.get("PROFSA_DUMP_DIR")
        if dump and getattr(self, "_dump_rows", None):
            seed = os.environ.get("PROFSA_DUMP_SEED", "0")
            os.makedirs(dump, exist_ok=True)
            with open(os.path.join(dump, f"ProFSA_v2_seed{seed}.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["pid", "pred", "y"])
                for n, p, y in self._dump_rows:
                    w.writerow([n, float(p), float(y)])
            logger.info(f"[dump] wrote {len(self._dump_rows)} per-pid preds -> {dump}/ProFSA_v2_seed{seed}.csv")
            self._dump_rows = []

        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.criterion.reset()
