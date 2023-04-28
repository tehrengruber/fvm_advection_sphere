# -*- coding: utf-8 -*-
import numpy as np
import os

from gt4py.next.program_processors.runners import gtfn_cpu, roundtrip
from gt4py.next.otf.compilation.build_systems import cmake
from gt4py.next.ffront.fbuiltins import float32, float64, int32, int64

run_gtfn_cached_cmake = gtfn_cpu.otf_compile_executor.CachedOTFCompileExecutor(
    name="run_gtfn_cached_cmake",
    otf_workflow=gtfn_cpu.workflow.CachedStep(step=gtfn_cpu.run_gtfn.otf_workflow.replace(
        compilation=gtfn_cpu.compiler.Compiler(
            cache_strategy=gtfn_cpu.cache.Strategy.PERSISTENT,
            builder_factory=cmake.CMakeFactory(cmake_build_type=cmake.BuildType.RELEASE)
        )),
    hash_function=gtfn_cpu.compilation_hash),
)

# backend = roundtrip.fencil_executor
# backend = gtfn_cpu.run_gtfn
backend = run_gtfn_cached_cmake

# set data types
precision = os.environ.get("FVM_PRECISION", "double").lower()
if precision == "double":
    float_type, int_type = float64, int64
elif precision == "single":
    float_type, int_type = float32, int32
else:
    raise ValueError("Only 'single' and 'double' precision are supported.")
