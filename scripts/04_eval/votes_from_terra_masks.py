"""Unpack a majority-vote mask written on the server (np.packbits of a (139, 512, 512) bool array, Test A chips in the
order of testA_common.TestA.names) into output/water_states/testA_translation_votes.npy, the vote file that every
Test A figure, table and number of the paper reads. The paper file is masks_P0_A.npy (P0, five seeds, >= 3 votes).
usage: votes_from_terra_masks.py <masks_*.npy> [--check]    --check only compares with the current vote file"""
import sys
import numpy as np

sys.path.insert(0, "/home/kk/jp_claude/RMSD/scripts/05_figures")
from testA_common import O, TestA  # noqa: E402

N = (139, 512, 512)
src = sys.argv[1]
U = np.unpackbits(np.load(src))[:np.prod(N)].reshape(N).astype(bool)
assert U.shape[0] == len(TestA().names)
dst = O + "testA_translation_votes.npy"
if "--check" in sys.argv:
    cur = np.load(dst).astype(bool)
    print("%s vs %s: %d differing pixels" % (src.split("/")[-1], dst, int((U != cur).sum())))
else:
    np.save(dst, U)
    print("wrote", dst, U.shape, "water share %.4f" % U.mean())
