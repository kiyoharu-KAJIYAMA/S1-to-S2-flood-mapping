"""Scene-matched control pool "MS" for REQUEST_terra_20261022_sameday_l1c_tau.md (follow-up, 2026-10-22).

MX matched SD's per-component chip counts but not its scene counts (SD 47 scenes, MX 103), so the SD-vs-MX
gap confounds "same-day" with "scene diversity". MS matches both: for every component it draws
k = (#SD scenes of that component) scenes from the full pool (any dt_days; drawn with probability
proportional to scene size, because SD scenes are the large ones and a uniform draw of 9 kuros2 scenes
almost never holds 328 chips), redrawing until their chips can
cover n = (#SD chips), then takes one chip from each chosen scene plus a random remainder.
fp is all same-day, so MS takes all of it (as SD and MX do). Same directory layout as build_pools.py;
nothing outside output/sameday/pools/MS is written.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
P = f"{ROOT}/processed"
OUT = f"{ROOT}/output/sameday/pools"
COMP = ["sen12", "s1f11", "c2s", "fp", "kuros2", "kurr", "sturm"]
FILES = ["trn16_component.csv", "trn16_chip_meta.csv", "trn16_X_VV_VH_RVI_raw.npy", "trn16_Y_MNDWI_NDVI_NDWI_raw.npy"]


def dt_bins(x):
    x = np.where(np.isnan(x), -1, x)
    s = pd.Series(x).map(lambda v: "0" if v == 0 else "1" if v == 1 else "2-3" if 0 < v <= 3 else ">3/NA")
    return {k: int(v) for k, v in s.value_counts().sort_index().items()}


def main():
    ci = pd.read_csv(f"{P}/trn16_component.csv")
    sp = pd.read_csv(f"{P}/trn16_split.csv")
    cm = pd.read_csv(f"{P}/trn16_chip_meta.csv", low_memory=False)
    assert len(ci) == len(sp) == len(cm) and (ci["idx"].to_numpy() == np.arange(len(ci))).all()
    pool = (ci["component"].isin(COMP).to_numpy() & (sp["split"] == "train").to_numpy()
            & ~(cm["water_frac_measured"].to_numpy() < 0.005))
    dt = pd.to_numeric(cm["dt_days"], errors="coerce").to_numpy()
    comp = ci["component"].to_numpy()
    scene = ci["scene"].to_numpy()
    idx = np.arange(len(ci))
    sd = pool & (dt == 0)

    rng = np.random.default_rng(0)
    out, info = [], {}
    for c in COMP:
        n = int((sd & (comp == c)).sum())
        if n == 0:
            continue
        k = int(pd.Series(scene[sd & (comp == c)]).nunique())
        cand = idx[pool & (comp == c)]
        if c == "fp":
            rows = cand
        else:
            sc = pd.Series(scene[cand]).value_counts()
            for _ in range(100000):
                pick = rng.choice(sc.index.to_numpy(), size=k, replace=False, p=(sc / sc.sum()).to_numpy())
                if sc[pick].sum() >= n:
                    break
            else:
                raise RuntimeError(f"no {k}-scene set covers {n} chips for {c}")
            inpick = cand[np.isin(scene[cand], pick)]
            first = np.array([rng.choice(inpick[scene[inpick] == s]) for s in pick])
            rest = rng.choice(np.setdiff1d(inpick, first), size=n - k, replace=False)
            rows = np.concatenate([first, rest])
        assert len(rows) == n and pd.Series(scene[rows]).nunique() == k, (c, len(rows), n)
        out.append(np.sort(rows))
        info[c] = {"n": n, "scenes": k, "scenes_shared_with_SD": int(len(set(scene[rows]) & set(scene[sd & (comp == c)]))),
                   "dt_days": dt_bins(dt[rows])}
    rows = np.sort(np.concatenate(out))
    assert len(rows) == 848 and pd.Series(scene[rows]).nunique() == 47, (len(rows), pd.Series(scene[rows]).nunique())

    d = f"{OUT}/MS"
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        if not os.path.lexists(f"{d}/{f}"):
            os.symlink(f"{P}/{f}", f"{d}/{f}")
    s2 = sp.copy()
    keep = np.zeros(len(s2), bool)
    keep[rows] = True
    s2.loc[~keep, "split"] = "sameday_excluded"
    s2.to_csv(f"{d}/trn16_split.csv", index=False)
    pd.DataFrame({"row": rows, "component": comp[rows], "scene": scene[rows], "dt_days": dt[rows]}).to_csv(f"{d}/rows.csv", index=False)
    chk = pd.read_csv(f"{d}/trn16_split.csv")
    n_train = int(((chk["split"] == "train") & ci["component"].isin(COMP) & ~(cm["water_frac_measured"] < 0.005)).sum())
    assert n_train == len(rows), n_train
    summ = {"n": int(len(rows)), "scenes": 47, "dt_days": dt_bins(dt[rows]),
            "overlap_chips_with_SD": int(sd[rows].sum()), "by_component": info}
    json.dump(summ, open(f"{OUT}/pool_MS_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
