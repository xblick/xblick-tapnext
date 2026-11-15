"""The node with the tracker's step on its third frame made slower by SLOW_S seconds (its first argument), for
tests/test_node.py: its queryable answers while the model works. It prints "slow step" when that step begins."""

import sys
import time

from xblick_tapnext import model, node

SLOW_S = float(sys.argv.pop(1))
step = model.SlotTracker.step
steps = [0]


def slow_step(self, *args, **kwargs):
    steps[0] += 1
    if steps[0] == 3:
        print("slow step", flush=True)
        time.sleep(SLOW_S)
    return step(self, *args, **kwargs)


model.SlotTracker.step = slow_step
sys.exit(node.main())
