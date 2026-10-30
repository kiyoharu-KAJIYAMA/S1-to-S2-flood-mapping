#!/bin/bash
# usage: eval_watcher.sh <tier> <run names...> ; waits until every run has an "end ... exit=0" line in the
# worker logs, then runs eval_tier<tier>.py on CPU (cores 128-191, nice 10, no GPU).
T=$1; shift
R=/lustre/home/kajiyama/rmsd; L=$R/output/l1mndwi0/logs
for r in "$@"; do
  until grep -q "name=$r exit=" $L/worker_gpu*.log 2>/dev/null; do sleep 60; done
  grep -q "name=$r exit=0" $L/worker_gpu*.log || { echo "$(date '+%F %T') $r failed, eval not started" >> $L/eval_watcher_tier$T.log; exit 1; }
done
echo "$(date '+%F %T') all done, eval tier $T" >> $L/eval_watcher_tier$T.log
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 RMSD_ROOT=$R
cd $R/scripts/12_l1mndwi0 && nice -n 10 taskset -c 128-191 ~/rmsd_terra/bin/python -u eval_tier$T.py > $L/eval_tier$T.log 2>&1
echo "$(date '+%F %T') eval tier $T exit=$?" >> $L/eval_watcher_tier$T.log
