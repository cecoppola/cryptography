[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          R E F   —   R E F E R E N C E   N T T   ( S I N G L E - P R I M E )                           ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat ref/README.md   or   less -R ref/README.md[0m

  ref is the reference-and-roadmap layer: single-prime Number-Theoretic
  Transform implementations that are correct, readable, and exhaustively
  host-tested. It is where every algorithm is proven on the CPU before the
  performance work happens in lib (the 4-prime CRT engine) on the MI300A.
  Unlike lib, every ref binary builds and runs on a plain CPU box with no
  ROCm/GPU — the .hip kernels here are cross-compiled for parity only.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  CONTENTS[0m

  ┌──────────────────────────┬──────────────────────────────────────────────┐
  │ [1;36mFile[0m                     │ [1;36mRole[0m                                         │
  ├──────────────────────────┼──────────────────────────────────────────────┤
  │ ntt_cpu.c                │ CT-DIT NTT, CPU reference                    │
  │ ntt_stockham.c           │ Stockham auto-sort NTT (distinct index math) │
  │ ntt_polymul.c            │ cyclic polynomial multiply driver            │
  │ ntt_polymul_negacyclic.c │ twisted (negacyclic) polynomial multiply     │
  │ ntt_mlkem.c              │ FIPS-203 ML-KEM 7-layer NTT (q=3329)         │
  │ ntt_bench.c              │ algorithm sweep over the 14-prime table      │
  │ ntt_moduli.h             │ curated 14-prime table + reductions          │
  │ ntt.h                    │ shared params/twiddle/transform API          │
  │ ntt_gpu*.hip             │ GPU kernels (gfx1030/gfx942 cross-compile)   │
  │ test_*.c                 │ host test suites (see make targets below)    │
  └──────────────────────────┴──────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  TESTING[0m

  All host tests are GPU-free and wired into `make check` at the repo root.
  Per-suite targets: test-curated, test-ntt-rigor, test-ntt-rigor-stok,
  test-polymul-integ, test-negacyc-integ, test-mlkem-kat, test-params-
  boundary. Meta: verify-nonvacuous (fault-injection proof), test-asan
  (ASAN+UBSan), coverage (gcov, 90% floor). The GPU kernels are verified
  on hardware via cross-verify / determinism (gfx-gated, not in `check`).
