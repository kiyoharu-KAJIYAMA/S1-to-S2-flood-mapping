#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation-axis redesign: assign train / holdout / excluded to all 4,875 trn16 chips.

Policy
  axis A (secondary, optical-based, conservative) ... Sen1Floods11 held out per event.
                                   The same C2S-MS events are removed in lockstep.
  axis B (primary, independent, purely optical)   ... some FloodPlanet regions held out.
                                   Labels are hand-drawn on 3.3 m PlanetScope without reference to SAR.
  leak removal                     ... FloodPlanet has regions that are the "same flood, same date"
                                   as Sen1Floods11 (measured: bbox overlap 92-100%). Any flood that
                                   goes to the evaluation side is always removed from training.

Output: processed/trn16_split.csv  (idx, dataset, event, split, reason)
Standard library only.
"""
import collections
import csv
import os

P = os.environ.get("RMSD_PROCESSED", "/home/kk/jp_claude/RMSD/processed")

# --- axis A: Sen1Floods11 floods held out per event ---
#   mix high/medium/low water fractions and spread continents (India=Asia, Somalia=Africa, Spain=Europe)
A_HOLDOUT = {"India", "Somalia", "Spain"}

# --- axis B: FloodPlanet held-out regions (those not sharing a flood with Sen1Floods11) ---
B_HOLDOUT = {"Cambodia", "Colombia", "Nepal", "Bangladesh"}

# --- FloodPlanet regions that are the same flood as Sen1Floods11/C2S (verified by measurement) ---
#   Bolivia is the same flood/date as the S1F11 eval (bolivia split) -> cannot stay in training
FP_SHARED_WITH_S1F11 = {"Ghana", "Nigeria", "Paraguay", "Somalia", "Spain", "Bolivia"}


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    cm = rd(f"{P}/trn16_chip_meta.csv")
    out = []
    for r in cm:
        ds, ev = r["dataset"], r["event"]
        split, reason = "train", ""

        if ds in ("s1f11", "c2s") and ev in A_HOLDOUT:
            split = "holdout_A"
            reason = f"axis A: Sen1Floods11 event '{ev}' held out (c2s linked)"

        elif ds == "fp":
            if ev in B_HOLDOUT:
                split, reason = "holdout_B", f"axis B: FloodPlanet region '{ev}' held out"
            elif ev == "Bolivia":
                split = "excluded"
                reason = ("same flood/date as the Sen1Floods11 'bolivia' eval split "
                          "(bbox overlap 26%, 2018-02-15)")
            elif ev in A_HOLDOUT:
                split = "excluded"
                reason = (f"same flood/date as axis-A holdout event '{ev}' "
                          f"(bbox overlap >=92%)")

        out.append(dict(idx=r["chip_index"], dataset=ds, event=ev,
                        split=split, reason=reason))

    dst = f"{P}/trn16_split.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "dataset", "event", "split", "reason"])
        w.writeheader()
        w.writerows(out)

    # ---------------- report ----------------
    cnt = collections.Counter(r["split"] for r in out)
    print(f"wrote {dst}  ({len(out)} rows)\n")
    print(f"{'split':12} {'chips':>6}")
    for k in ["train", "holdout_A", "holdout_B", "excluded"]:
        print(f"{k:12} {cnt[k]:6}")
    print(f"{'TOTAL':12} {sum(cnt.values()):6}")

    print("\n--- breakdown of chips removed from training ---")
    for k in ["holdout_A", "holdout_B", "excluded"]:
        sub = [r for r in out if r["split"] == k]
        by = collections.Counter((r["dataset"], r["event"]) for r in sub)
        print(f"\n[{k}]  {len(sub)} chips")
        for (d, e), n in sorted(by.items(), key=lambda x: (-x[1], x[0])):
            print(f"   {d:8} {e:16} {n:5}")

    print("\n--- remaining training data (by component) ---")
    tr = collections.Counter(r["dataset"] for r in out if r["split"] == "train")
    before = collections.Counter(r["dataset"] for r in out)
    print(f"{'dataset':10} {'before':>7} {'after':>7} {'diff':>7}")
    for k in ["sen12", "s1f11", "c2s", "fp", "kuros2", "gfd", "kurr", "sturm"]:
        print(f"{k:10} {before[k]:7} {tr[k]:7} {tr[k]-before[k]:+7}")
    print(f"{'TOTAL':10} {sum(before.values()):7} {sum(tr.values()):7} "
          f"{sum(tr.values())-sum(before.values()):+7}")


if __name__ == "__main__":
    main()
