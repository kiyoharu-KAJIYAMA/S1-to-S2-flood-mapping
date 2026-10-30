#!/usr/bin/env python3
"""Cross-fold summary (request section 6, last paragraph): one row per fold, the
main quantities of 6-1, the share of recovery within 50 m, the VV median of
unrecovered water, for the t18 majority and for cur-s, plus the calibrated
thresholds and the seed spread. Folds whose evaluation is missing are skipped.

Output: output/loco/summary.json and output/loco/summary.md
"""
import json
import os

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
OUT = f"{ROOT}/output/loco"
FOLDS = ["Official64", "USA", "India", "Paraguay", "Ghana", "Sri-Lanka", "Mekong", "Spain",
         "Pakistan", "Somalia", "Nigeria", "Bolivia"]
PAPER_HOLD = {"Bolivia", "India", "Somalia", "Spain"}


def row(d, det):
    s = d["detectors"][det]
    a = s["6-1"]
    vv = s["6-4"]["unrecovered_water"]["VV"]
    return dict(iou=a["iou"], precision=a["precision"], recall=a["recall"],
                land_fa=a["land_false_alarm"], missed=a["missed_share"], recovery=a["recovery"],
                within_50m=s["6-3"]["within_50m"], unrec_vv_median=vv[1] if vv else None)


def main():
    res = {"folds": {}}
    lines = ["| Fold | n_test | n_calib | rule IoU | t18 IoU | t18 recovery | t18 within 50 m | unrecovered VV | cur IoU | cur recovery | cur within 50 m | IoU diff (cur-t18) | recovery diff | tau (s1/s2/s3) | rule (b) VV/VH |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for f in FOLDS:
        p = f"{OUT}/eval/{f}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        t, c, r = row(d, "t18_majority"), row(d, "cur_majority"), row(d, "rule_a")
        taus = [d["calibration"]["t18"][k]["tau"] for k in d["runs"]]
        rb = d["calibration"]["rule_b"]
        res["folds"][f] = dict(n_test=d["n_test"], n_calib=d["n_calib"],
                               cur_saw_this_country=f not in PAPER_HOLD and f != "Official64",
                               rule_a=r, t18=t, cur=c,
                               diff_cur_minus_t18=dict(iou=round(c["iou"] - t["iou"], 4),
                                                       recovery=round(c["recovery"] - t["recovery"], 4)),
                               taus=taus, rule_b=[rb["lee_vv_db"], rb["lee_vh_db"]],
                               seed_spread=d["seed_spread"])
        mark = "" if f in PAPER_HOLD or f == "Official64" else "†"
        lines.append(f"| {f}{mark} | {d['n_test']} | {d['n_calib']} | {r['iou']:.4f} | {t['iou']:.4f} | "
                     f"{t['recovery']:.4f} | {t['within_50m']:.3f} | {t['unrec_vv_median']} | "
                     f"{c['iou']:.4f} | {c['recovery']:.4f} | {c['within_50m']:.3f} | "
                     f"{c['iou'] - t['iou']:+.4f} | {c['recovery'] - t['recovery']:+.4f} | "
                     f"{'/'.join(f'{x:.2f}' for x in taus)} | {rb['lee_vv_db']:.0f}/{rb['lee_vh_db']:.0f} |")
    lines.append("")
    lines.append("† cur-s saw this country in training (the difference is the gain from having seen it). cur-s did not see the 64 Official64 chips.")
    if res["folds"]:
        vals = [v["t18"]["recovery"] for v in res["folds"].values()]
        w50 = [v["t18"]["within_50m"] for v in res["folds"].values()]
        res["range"] = dict(recovery_min=min(vals), recovery_max=max(vals),
                            within_50m_min=min(w50), within_50m_max=max(w50), n_folds=len(vals))
        lines.append(f"\nRecovery range: {min(vals):.3f}-{max(vals):.3f}, share within 50 m: {min(w50):.3f}-{max(w50):.3f}"
                     f" ({len(vals)} folds)")
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2, ensure_ascii=False)
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {OUT}/summary.json, summary.md ({len(res['folds'])} folds)")


if __name__ == "__main__":
    main()
