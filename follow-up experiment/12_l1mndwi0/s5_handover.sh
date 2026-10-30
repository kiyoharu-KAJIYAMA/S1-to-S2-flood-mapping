#!/bin/bash
# Wait for the bidirectional P0-s5 (python pid 2512939, kept as the 5th bidirectional reference seed) to exit,
# move its run dir / log to runs_bidir / logs/bidir, then start the one-direction worker on GPU 0.
O=/lustre/home/kajiyama/rmsd/output/l1mndwi0; D=/lustre/home/kajiyama/rmsd/scripts/12_l1mndwi0
while kill -0 2512939 2>/dev/null; do sleep 20; done
sleep 5
mv $O/runs/P0-s5 $O/runs_bidir/ && mv $O/logs/train_P0-s5.log $O/logs/train_P0-s5.pid $O/logs/bidir/ && mv $O/logs/worker_gpu0.log $O/logs/bidir/worker_gpu0.log
echo "$(date '+%F %T') bidir P0-s5 moved; starting one-direction worker on GPU 0" >> $O/logs/s5_handover.log
cd $D && nohup ./worker_onedir.sh 0 0-31 $D/queue_onedir.txt > $O/logs/worker_onedir0.out 2>&1 &
echo $! > $O/logs/worker_onedir0.pid
