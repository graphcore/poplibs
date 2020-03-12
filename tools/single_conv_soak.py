#!/usr/bin/env python3
# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import os
import subprocess
import sys

if len(sys.argv) != 2:
    raise Exception(f"{sys.argv[0]} expects a single argument, the random seed")

seed = sys.argv[1]
script = os.path.join(sys.path[0], "single_conv_layer_random.py")

rc = subprocess.call([
    script,
    "--n", "1",
    "--seed", seed,
    "--device-type", "IpuModel",
    "--tiles-per-ipu", "1216",
])
sys.exit(rc)
