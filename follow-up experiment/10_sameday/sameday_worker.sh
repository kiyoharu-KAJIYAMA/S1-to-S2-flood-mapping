#!/bin/bash
# usage: sameday_worker.sh <gpu_index(PCI)> <cpuset> <queue_file>
# REQUEST_terra_20261022_sameday_l1c_tau.md: retrain the paper translation model (cur-s* settings =
# train_variant.py defaults, in_mask 1,1,1) on a restricted pool. Queue line: "<run_name> <arm> <train args...>",
# e.g. "SD-s1 SD --seed 1". --datapath is output/sameday/pools/<arm>/ (split.csv rewritten there, arrays
# are symlinks to processed/), outputs go to output/sameday/runs/<run_name>/ and output/sameday/logs/.
# Same operating rules as scripts/09_fig5/train_worker.sh: one run per GPU, OMP=1, nice -n 10,
# python PID written to logs/train_<run>.pid (kill by PID, never by name), mkdir lock on the queue.
set -u
R=${RMSD_ROOT:-/lustre/home/kajiyama/rmsd}
GPU=$1; CPUS=$2; Q=$3
S=$R/output/sameday
LOCK=$Q.lock
mkdir -p $S/runs $S/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RMSD_ROOT=$R
cd $R/scripts/06_ablation || exit 1
while true; do
  until mkdir "$LOCK" 2>/dev/null; do sleep 2; done
  LINE=$(grep -v '^\s*$' "$Q" | head -1)
  if [ -n "$LINE" ]; then
    grep -v -x -F "$LINE" "$Q" > "$Q.tmp"; mv -f "$Q.tmp" "$Q"
  fi
  rmdir "$LOCK"
  [ -z "$LINE" ] && { echo "queue empty $(date '+%F %T') gpu=$GPU" >> $S/logs/worker_gpu$GPU.log; break; }
  NAME=$(echo "$LINE" | awk '{print $1}'); ARM=$(echo "$LINE" | awk '{print $2}')
  ARGS=$(echo "$LINE" | cut -d' ' -f3-)
  LOG=$S/logs/train_${NAME}.log
  echo "start $(date '+%F %T') gpu=$GPU cpus=$CPUS name=$NAME arm=$ARM args=$ARGS" | tee -a $S/logs/worker_gpu$GPU.log > $LOG
  nice -n 10 taskset -c $CPUS ~/rmsd_terra/bin/python -u train_variant.py \
    --rootpath $S --datapath $S/pools/$ARM --outpath $S/runs \
    --date_out $NAME --gpu 0 $ARGS >> $LOG 2>&1 &
  PYPID=$!
  echo $PYPID > $S/logs/train_${NAME}.pid
  echo "  python pid=$PYPID" >> $S/logs/worker_gpu$GPU.log
  wait $PYPID; RC=$?
  echo "end   $(date '+%F %T') gpu=$GPU name=$NAME exit=$RC" >> $S/logs/worker_gpu$GPU.log
done
