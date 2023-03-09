# -*- coding: utf-8 -*-
import numpy as np
import os

from functional.program_processors.runners import gtfn_cpu, roundtrip
from functional.ffront.fbuiltins import float32, float64, int32, int64

# backend = roundtrip.fencil_executor
backend = gtfn_cpu.run_gtfn

# set data types
precision = os.environ.get("FVM_PRECISION", "double").lower()
if precision == "double":
    float_type, int_type = float64, int64
elif precision == "single":
    float_type, int_type = float32, int32
else:
    raise ValueError("Only 'single' and 'double' precision are supported.")
