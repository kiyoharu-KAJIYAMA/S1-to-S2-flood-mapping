#!/bin/bash
# Start the Tier 0/1/2 eval watchers once the bidirectional worker_gpu0.log has been moved away (s5_handover.sh),
# so that its old "name=P0-s1 exit=0" lines cannot be mistaken for the one-direction runs.
L=/lustre/home/kajiyama/rmsd/output/l1mndwi0/logs; D=/lustre/home/kajiyama/rmsd/scripts/12_l1mndwi0
until grep -q "bidir P0-s5 moved" $L/s5_handover.log 2>/dev/null; do sleep 20; done
cd $D
nohup ./eval_watcher.sh 0 P0-s1 P0-s2 P0-s3 P0-s4 P0-s5 L1w2xy-s1 L1w2xy-s2 L1w2xy-s3 > /dev/null 2>&1 & echo $! > $L/eval_watcher_tier0.pid
nohup ./eval_watcher.sh 1 P0-v7c-s1 P0-v7c-s2 P0-v7c-s3 P0-v7b-s1 P0-v7b-s2 P0-v7b-s3 P0-v7a-s1 P0-v7a-s2 P0-v7a-s3 > /dev/null 2>&1 & echo $! > $L/eval_watcher_tier1.pid
nohup ./eval_watcher.sh 2 P0-s1 P0-s2 P0-s3 P0-s4 P0-s5 $(for k in 1 2 3; do for a in SD MS L1 MX NC; do echo -n "P0-$a-s$k "; done; done) > /dev/null 2>&1 & echo $! > $L/eval_watcher_tier2.pid
echo "$(date '+%F %T') watchers started" >> $L/s5_handover.log
