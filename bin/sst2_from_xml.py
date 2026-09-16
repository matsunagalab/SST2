#!/usr/bin/env python3
"""Run one SST2 walker from system.xml + topology.pdb (+ state.xml).
See SST2.driver for the arguments."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))
from SST2.driver import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
