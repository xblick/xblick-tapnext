"""The node with torch.compile made to fail, for tests/test_node.py: the node compiles by default, says so at its
start, and a failed compilation ends it with one line (never an uncompiled network instead)."""

import sys

import torch

from xblick_tapnext import node


def failing_compile(*args, **kwargs):
    raise RuntimeError("no compiler here")


torch.compile = failing_compile
sys.exit(node.main())
