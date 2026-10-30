#!/bin/bash
# usage: gworker.sh <gpu_index(PCI)> <cpuset> <queue_file>
# GPU queue for REQUEST_terra_20261026_l1only_mndwi0.md (max 4 GPUs, requester 2026-10-26); one-direction (SAR->MSI) training
# with scripts/09_onedir/train_onedir.py (requester decision 2026-10-26). Otherwise identical to worker.sh.
# Queue line = absolute path of a config JSON (copy of a t18-Paper config); every non-empty key is passed to
# train_variant.py. Log/pid go to <outpath>/../logs/. Rules as 11_loss_ablation/worker.sh: one run per GPU,
# OMP=1, nice -n 10, kill by the python PID in the .pid file, mkdir lock on the queue.
set -u
GPU=$1; CPUS=$2; Q=$3; LOCK=$Q.lock
R=/lustre/home/kajiyama/rmsd
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RMSD_ROOT=$R
cd $R/scripts/06_ablation || exit 1
WL=$R/output/l1mndwi0/logs/worker_gpu$GPU.log
while true; do
  until mkdir "$LOCK" 2>/dev/null; do sleep 2; done
  CFG=$(grep -v '^\s*$' "$Q" | head -1)
  if [ -n "$CFG" ]; then grep -v -x -F "$CFG" "$Q" > "$Q.tmp"; mv -f "$Q.tmp" "$Q"; fi
  rmdir "$LOCK"
  [ -z "$CFG" ] && { echo "queue empty $(date '+%F %T') gpu=$GPU" >> $WL; break; }
  read NAME LOGD ARGS < <(~/rmsd_terra/bin/python -I -c 'import json,sys,os
c=json.load(open(sys.argv[1])); a=" ".join(f"--{k} {str(v).lower() if isinstance(v,bool) else v}" for k,v in c.items() if v != "")
print(c["date_out"], os.path.join(os.path.dirname(c["outpath"]), "logs"), a)' "$CFG")
  mkdir -p $LOGD; LOG=$LOGD/train_${NAME}.log
  echo "start $(date '+%F %T') gpu=$GPU cpus=$CPUS name=$NAME cfg=$CFG" | tee -a $WL > $LOG
  echo "args $ARGS" >> $LOG
  nice -n 10 taskset -c $CPUS ~/rmsd_terra/bin/python -u $R/scripts/09_onedir/train_onedir.py $ARGS >> $LOG 2>&1 &
  PYPID=$!
  echo $PYPID > $LOGD/train_${NAME}.pid
  echo "  python pid=$PYPID" >> $WL
  wait $PYPID; RC=$?
  echo "end   $(date '+%F %T') gpu=$GPU name=$NAME exit=$RC" >> $WL
done
