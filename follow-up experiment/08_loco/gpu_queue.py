#!/usr/bin/env python3
"""GPU queue for the trn18 runs (request section 8.4).

Holds the full run list, measures the free VRAM of every GPU whenever a slot is
needed, and starts the next run wherever it fits. Per-GPU capacity is
floor((free - margin) / (vram_per_run + margin)); a GPU whose utilisation stays
above 90% from other people's jobs gets at most one run. RAM is re-checked so
that runs * rss + 20 GB stays below the free memory. Everything the queue does is
logged to output/loco/queue.log (start/end time, GPU, exit code).

Usage: queue.py --vram_gb <per run> --rss_gb <per run> [--runs list.txt]
       [--max_total N] [--dry]
The run list file has one "<name> <fold> <seed>" per line; without it the full
Tier 1 + Tier 1c order of section 4.3 is used.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.environ.get("RMSD_ROOT", "/lustre/home/kajiyama/rmsd")
OUT = f"{ROOT}/output/loco"
PY = os.path.expanduser("~/rmsd_terra/bin/python")
TRAIN = f"{ROOT}/scripts/06_ablation/train_variant.py"
FOLDS = ["Official64", "USA", "India", "Paraguay", "Ghana", "Sri-Lanka", "Mekong", "Spain",
         "Pakistan", "Somalia", "Nigeria", "Bolivia"]
NCPU = os.cpu_count() or 256


def log(msg):
    line = f"{datetime.now():%F %T} {msg}"
    print(line, flush=True)
    with open(f"{OUT}/queue.log", "a") as f:
        f.write(line + "\n")


def gpu_state():
    """[(index, free_mb, util%, own_runs)] from nvidia-smi. own_runs counts every
    compute process of this user on the GPU, including runs the queue did not
    start, so a restart never stacks on top of jobs already running."""
    txt = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    me = os.getuid()
    own = {}
    for line in apps.strip().splitlines():
        pid, uuid = [x.strip() for x in line.split(",")]
        try:
            if os.stat(f"/proc/{pid}").st_uid == me:
                own[uuid] = own.get(uuid, 0) + 1
        except OSError:
            pass
    out = []
    for line in txt.strip().splitlines():
        i, uuid, used, total, util = [x.strip() for x in line.split(",")]
        out.append((int(i), int(total) - int(used), int(util), own.get(uuid, 0)))
    return out


def free_ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def default_runs(seeds):
    return [(f"t18-{f}-s{s}", f, s) for f in FOLDS for s in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram_gb", type=float, required=True, help="VRAM of one run (GB)")
    ap.add_argument("--rss_gb", type=float, required=True, help="RSS of one run (GB)")
    ap.add_argument("--margin_gb", type=float, default=2.0)
    ap.add_argument("--runs", default="", help="file with '<name> <fold> <seed>' lines")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max_total", type=int, default=64, help="hard cap on simultaneous runs")
    ap.add_argument("--max_per_gpu", type=int, default=64,
                    help="cap per GPU; a saturated GPU gains little from more than 2-3 runs")
    ap.add_argument("--busy_util", type=int, default=90)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.runs:
        runs = [tuple(l.split()[:3]) for l in open(args.runs) if l.strip()]
        runs = [(n, f, int(s)) for n, f, s in runs]
    else:
        runs = default_runs([int(s) for s in args.seeds.split(",")])
    def is_done(name):
        lf = f"{OUT}/logs/train_{name}.log"
        return os.path.exists(lf) and any(l.startswith("done ") for l in open(lf, errors="ignore"))

    skipped = [r[0] for r in runs if is_done(r[0])]
    pending = [r for r in runs if not is_done(r[0])]
    if skipped:
        log(f"already done, skipped: {skipped}")
    log(f"queue start: {len(pending)} runs, vram/run {args.vram_gb} GB, rss/run {args.rss_gb} GB")
    os.makedirs(f"{OUT}/runs", exist_ok=True)
    os.makedirs(f"{OUT}/logs", exist_ok=True)

    active = {}  # pid -> (name, gpu, Popen, start)
    util_hist = {}
    while pending or active:
        # reap finished runs
        for pid, (name, gpu, proc, t0) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                log(f"end   {name} gpu={gpu} exit={rc} {(time.time() - t0) / 60:.1f} min")
                del active[pid]
        # capacity per GPU
        state = gpu_state()
        cap, own = {}, {}
        for i, free_mb, util, mine_smi in state:
            util_hist.setdefault(i, []).append(util)
            util_hist[i] = util_hist[i][-5:]
            # a run spends its first minutes in preprocessing and is not yet on the
            # GPU, so count the queue's own children as well as what nvidia-smi sees
            mine_q = sum(1 for _, g, _, _ in active.values() if g == i)
            mine = max(mine_smi, mine_q)
            own[i] = mine
            n = int((free_mb / 1024 - args.margin_gb) // (args.vram_gb + args.margin_gb)) + mine
            busy = len(util_hist[i]) >= 3 and min(util_hist[i]) > args.busy_util and mine == 0
            if busy:
                n = min(n, 1)
            cap[i] = max(min(n, args.max_per_gpu), 0)
        ram_ok = int((free_ram_gb() - 20) // args.rss_gb)
        started = 0
        for i, _, _, _ in sorted(state, key=lambda t: -t[1]):
            mine = own[i]
            while pending and mine < cap[i] and len(active) < min(args.max_total, ram_ok):
                name, fold, seed = pending.pop(0)
                threads = max(4, NCPU // max(len(active) + 1, 1))
                env = dict(os.environ, RMSD_ROOT=ROOT, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           CUDA_VISIBLE_DEVICES=str(i), OMP_NUM_THREADS=str(min(threads, 32)),
                           MKL_NUM_THREADS=str(min(threads, 32)),
                           OPENBLAS_NUM_THREADS=str(min(threads, 32)))
                cmd = ["nice", "-n", "10", PY, "-u", TRAIN,
                       "--rootpath", f"{OUT}/root_{fold}",
                       "--datapath", f"{OUT}/root_{fold}/processed",
                       "--outpath", f"{OUT}/runs", "--date_out", name, "--gpu", "0",
                       "--data_prefix", "trn18", "--seed", str(seed)]
                if args.dry:
                    log(f"DRY   {name} -> gpu={i}")
                    continue
                lf = open(f"{OUT}/logs/train_{name}.log", "w")
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                        cwd=f"{ROOT}/scripts/06_ablation")
                active[proc.pid] = (name, i, proc, time.time())
                mine += 1
                started += 1
                log(f"start {name} gpu={i} pid={proc.pid} threads={min(threads, 32)} "
                    f"(active {len(active)}, cap gpu{i}={cap[i]}, ram_ok={ram_ok})")
                time.sleep(20)  # let the new process allocate before measuring again
        if args.dry and not active:
            break
        time.sleep(args.poll if not started else 5)
    log("queue empty")


if __name__ == "__main__":
    main()
