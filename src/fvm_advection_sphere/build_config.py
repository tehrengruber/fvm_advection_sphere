from functional.program_processors.runners import gtfn_cpu, roundtrip

# backend = roundtrip.fencil_executor
backend = gtfn_cpu.run_gtfn
