"""Reproduction gate: rule and 5-run majority translation on Test A and B.
Writes output/ablation_recovery/gate.json and the baseline evaluations that the
later experiments reuse (rule, dilated rule, translation)."""
import json
import sys

import numpy as np

import common as C

TARGET = {
    "B": {"rule": dict(iou=0.580, precision=0.873, recall=0.634, missed=0.366),
          "unet": dict(iou=0.687, precision=0.883, recall=0.755, recovery=0.341)},
    "A": {"rule": dict(iou=0.401, precision=0.600, recall=0.548, missed=0.452),
          "unet": dict(iou=0.496, precision=0.711, recall=0.621, recovery=0.286)},
}
TOL = 0.003


def main():
    dev = C.device()
    print("device", dev, flush=True)
    gate = {}
    ok_all = True
    for test in ("B", "A"):
        T = C.load(test)
        print(f"Test {test}: chips {len(T.names)}  valid px {int(T.valid.sum()):,}  "
              f"gt {int(T.gt.sum()):,}", flush=True)
        res = {}
        res["rule"] = C.evaluate(T.rule, T)
        for r in (10, 20, 30, 50, 100):
            res[f"rule_dilate_{r}m"] = C.evaluate(T.rule | (T.D <= r), T)
        U = C.votes(T.X, test, dev)
        np.save(f"{C.OUT}/votes_translation_{test}.npy", U)
        res["translation"] = C.evaluate(U, T)
        for k, v in res.items():
            print(f"  {k:18} {C.brief(v)}", flush=True)
        C.save_json(res, f"{C.OUT}/baselines_{test}.json")
        g = {}
        tr, tu = TARGET[test]["rule"], TARGET[test]["unet"]
        got_r = dict(iou=res["rule"]["overall"]["iou"], precision=res["rule"]["overall"]["precision"],
                     recall=res["rule"]["overall"]["recall"], missed=res["rule"]["missed_share"])
        got_u = dict(iou=res["translation"]["overall"]["iou"],
                     precision=res["translation"]["overall"]["precision"],
                     recall=res["translation"]["overall"]["recall"],
                     recovery=res["translation"]["recovery"]["rate"])
        for nm, got, tgt in (("rule", got_r, tr), ("translation", got_u, tu)):
            g[nm] = {k: dict(target=tgt[k], got=got[k], pass_=abs(got[k] - tgt[k]) <= TOL)
                     for k in tgt}
            ok_all &= all(x["pass_"] for x in g[nm].values())
        gate[test] = g
    gate["pass"] = bool(ok_all)
    C.save_json(gate, f"{C.OUT}/gate.json")
    print(json.dumps(gate, indent=1))
    print("GATE", "PASS" if ok_all else "FAIL", flush=True)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
