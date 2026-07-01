[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                 A P P   —   C O M P U T E _ E   A P P L I C A T I O N                                  ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat app/README.md   or   less -R app/README.md[0m

  app/compute_e is the application layer: it computes Euler's number e to
  arbitrary precision and is the end-to-end consumer of the lib CRT-NTT
  engine. e is summed as the series e = sum_{k>=0} 1/k! evaluated by
  binary splitting; the large bigint multiplies that the recursion and the
  final scaling division need are dispatched to lib (GPU NTT above the
  schoolbook threshold, CPU schoolbook below). It doubles as a real-
  workload correctness and performance test of the whole stack.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  CONTENTS[0m

  ┌────────────────────────┬────────────────────────────────────────────┐
  │ [1;36mFile[0m                   │ [1;36mRole[0m                                       │
  ├────────────────────────┼────────────────────────────────────────────┤
  │ main.c                 │ driver: split -> 10^D -> divide -> decimal │
  │ binary_split.c         │ binary-splitting recursion for sum 1/k!    │
  │ mem_pool.c             │ scratch-buffer pool for the recursion      │
  │ ntt_stub.c             │ host NTT stub for the CPU-only test build  │
  │ test_binary_split.c    │ binary_split vs independent radix-2^30 ref │
  │ test_ntt_kernel.hip    │ GPU NTT known-answer tests (gfx-gated)     │
  │ test_polymul_sweep.hip │ GPU full-pipeline differential (gfx-gated) │
  └────────────────────────┴────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  TESTING[0m

  Host (GPU-free, in `make check`):  `make test-binsplit` runs the 26-cell
  binary_split adversarial test; `make test-host` builds a schoolbook-only
  compute_e and diffs its output against the digits of e (OEIS A001113).
  GPU (gfx-gated): test_ntt_kernel and test_polymul_sweep run on hardware
  via scripts/gpu_run.sh. Build/run on the MI300A requires the modules in
  scripts/setup_mi300a.sh.
