[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                               N T T   /   M I 3 0 0 A   —   A G E N T   G U I D A N C E                                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat CLAUDE.md   or   less -R CLAUDE.md[0m

  This file is loaded into the agent context every session. Read it before
  any task. The five hard rules at the top supersede every other heuristic.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. HARD RULES (formerly RULES.md — supersede every other instruction)[0m

  ┌───┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ [1;36m#[0m │ [1;36mRule[0m                                                                                                                                    │
  ├───┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 1 │ Be very token-efficient. Prefer CPU/memory over tokens. Run more commands rather than reading more bytes when it saves tokens.          │
  │ 2 │ Code comments must accurately reflect the current code. Update them when code changes. When reviewing files, audit comment correctness. │
  │ 3 │ Project docs must reflect current state. Update ROADMAP.md and any affected MD after each unit completes.                               │
  │ 4 │ Terminal output must be neat, elegant, fixed-width tables — no raw dumps.                                                               │
  │ 5 │ Rely on testing/benchmarking. Persist results in files; sweep parameter ranges.                                                         │
  │ 6 │ Completeness checks (audits/inventories/coverage) need a second independent pass before done: re-derive, list gaps, fix, accept.        │
  └───┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. PROJECT GOAL[0m

  Highest-performance, elegant C / HIP implementation of the modular Number-
  Theoretic Transform that exploits the AMD MI300A APU. Cross-compiled from a
  development workstation. The project deploys on the MI300A target only;
  6900 XT is dev hardware.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. KEY CONSTRAINTS[0m

  ┌────────────────────┬────────────────────────────────────────────────────────────────┐
  │ [1;36mConstraint[0m         │ [1;36mDetail[0m                                                         │
  ├────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Language           │ C only. No C++ in kernels or host. .hip files use __restrict__ │
  │ Target chip        │ MI300A (CDNA3); wave 64; unified HBM3                          │
  │ Dev sequence       │ 5950X CPU → 5950X+6900XT (dev) → MI300A (target)               │
  │ Runtime parameters │ n, q, ω, block size, algorithm — all CLI / config              │
  │ Hardware queries   │ hipGetDeviceProperties at startup; no hardcoded constants      │
  │ Modularity         │ CPU and GPU kernels share the same public signature            │
  └────────────────────┴────────────────────────────────────────────────────────────────┘

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. SESSION-START PROTOCOL[0m

  1. A fresh agent: read AGENT_BRIEF.md first (dense one-file orientation —
     goal, state, mental model, objectives, landmines, key data, pointers).
     Then ROADMAP.md — it lists open work, MI300A acceptance criteria,
     and current phase status. README.md gives the build/run quickstart;
     ARCHITECTURE.md and PERFORMANCE.md are the design and baseline references.
  2. Detect hardware: rocminfo 2>/dev/null | grep -c "gfx".
       count > 0  →  pick a task from ROADMAP §3 that the present GPU can drive.
       count = 0  →  pick a CPU/cross-compile task from ROADMAP §3 / §5.
  3. Use Grep / Glob for symbol search; use Read only when about to edit.
  4. After each unit: update ROADMAP.md, verify build clean, then stop or compact.
  5. GPU runs: `make check` (host gate, ~5 min, GPU-free) invokes coverage /
     test_asan / verify_nonvacuous / cpu_testbench, which use `make clean-host`
     (root bin/ only) so the host gate doesn't wipe lib/app GPU binaries — but
     plain `make clean` does. For a dev-box GPU run, build the dev binary
     (e.g. `make -C app/compute_e dev`) and launch it through scripts/gpu_run.sh
     (1300 MHz pin, display-yield, S-guards). NOTE (2026-06-30): the old
     check-gpu / check-all / gpu_session.sh / group_*.sh orchestration is
     dev-box-only and now lives in archive/scripts_devbox/ (gpu_run.sh + its
     deps were restored to scripts/ as local-only, gitignored).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  5. TOOL DISCIPLINE[0m

  ┌───────────────────────┬────────────────────────────────────────┐
  │ [1;36mTask[0m                  │ [1;36mTool to use[0m                            │
  ├───────────────────────┼────────────────────────────────────────┤
  │ Search file contents  │ Grep tool                              │
  │ Read a file           │ Read tool                              │
  │ Find files by pattern │ Glob tool                              │
  │ Edit a file           │ Edit tool                              │
  │ Bash reserved for     │ build, compile, run, profile, git only │
  └───────────────────────┴────────────────────────────────────────┘

  For programmatic edits to source files (corruption sweeps, batch
  replaces, fault-injection harnesses), use Python with literal-string
  replace (str.replace / Path.write_text) rather than sed regex. sed
  treats [, ], *, ., \, $, ^ in C/HIP code as regex metacharacters and
  silently fails to match — caught here by the verify_nonvacuous.sh sed
  bug on the twiddle line `tw[k] = ... % p->q;` (sed read `[k]` as a
  character class, no replacement happened, and the test mis-reported
  every suite as VACUOUS). The Python rewrite using str.replace is the
  current canonical pattern (scripts/verify_nonvacuous.sh).

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  6. MD-FORMATTING MANDATE[0m

  Canonical MD files use the ANSI-boxed terminal style (see ARCHITECTURE.md):
  ANSI-coloured boxed titles, ANSI-coloured section headers, fixed-width
  Unicode tables (use the tablize helper for auto-sized columns), dividers
  between sections, and a "View with: cat / less -R" hint near the top.

  EXCEPTIONS — plain / GitHub-flavored markdown (NOT ANSI):
    - README.md — the GitHub repo landing page; must render on github.com, so
      it uses GitHub-flavored markdown (headings, pipe tables, fenced code).
      Do NOT convert it back to the ANSI box style.
    - AGENT_BRIEF.md, DEPLOYMENT_DIAGNOSTIC.md — agent-executed docs; plain
      markdown so the ANSI escapes are not token-noise to an LLM.

  CANONICAL DOCS ARE HAND-MAINTAINED. ARCHITECTURE.md, PERFORMANCE.md,
  ROADMAP.md, STATUS.md, and the per-layer lib*/README.md files are the single
  source of truth and are edited IN PLACE with targeted edits that preserve
  their ANSI/box structure. Do NOT keep a parallel generator script for an
  existing doc — that creates two drifting sources (the mistake this rule retires).

  When CREATING A NEW boxed MD file from scratch, use ./pretty_md.py
  (local-only, gitignored; box/hint/divider/section/tablize + colour helpers,
  emits real ESC bytes — replicate if absent) via a one-shot script,
  then maintain the resulting .md by direct edit thereafter. Python is
  permitted for MD generation only. Plain Markdown is acceptable ONLY
  for sections that are non-prose data (e.g. embedded code blocks,
  tables of citations).

  Reference: ~/.claude/CLAUDE.md has the global enforcement rule.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  7. COMPACT INSTRUCTIONS[0m

  [1;32mPreserve[0m when compacting:
    - ROADMAP.md open tasks + the in-progress unit
    - API signatures already implemented (from lib*/README.md + headers)
    - Active build errors / test failures under investigation
    - Newly discovered constraints not yet captured in ARCHITECTURE.md

  [1;31mDiscard[0m when compacting:
    - exploratory investigation results
    - superseded approaches
    - verbose tool output
    - any context already captured in the canonical docs or source files

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  8. PROJECT FILE INDEX[0m

  Layer layout (renamed 2026-06-30):
    lib/      THE deliverable — 4-prime CRT-NTT engine for the MI300A K4 node.
              Raw polynomial pipeline (main.hip -> crt_ntt, OpenSHMEM-
              distributed) + BigInt-multiply API (arith/) + on-target tests.
    app/      applications/demonstrators that consume lib (app/compute_e).
    ref/      single-prime CPU reference NTT; proving ground, stays in repo,
              NOT shipped to MI300A.
    archive/  retired/dev-only material kept for provenance — docs/ (history),
              lib_dead/ (transfer_gpu.hip retired engine, cswitch, bench_g9),
              scripts_devbox/ (6900XT GPU-safety), tools/, logs/.

  MD files (tracked canonical set):
    README.md          project entry, build/run quickstart
    ARCHITECTURE.md    layered design, APIs, moduli, CRT engine, MI300A optim
    PERFORMANCE.md     current measured baseline per hardware target
    ROADMAP.md         phase status, open MI300A tasks, future work
    CLAUDE.md          this file (agent guidance)
    ref/README.md  lib/README.md  app/README.md   per-layer overviews

  Tooling docs (tracked):
    scripts/        kept = host gates + MI300A bring-up: check.sh, ci.sh,
                    coverage.sh, coverage_full.sh, test_asan.sh, cpu_testbench.sh,
                    verify_nonvacuous.sh, test_compute_e.sh, setup_mi300a.sh,
                    mi300a_first_run.sh. (Dev-box GPU-safety scripts moved to
                    archive/scripts_devbox/.)

  Local-only (kept on the dev box, gitignored — NOT in a fresh clone):
    mi300a_environment_0509.txt  Cray PE module set (also embedded in
                                 scripts/setup_mi300a.sh + README build-env section)
    pretty_md.py                 MD-formatting helper (see §6; replicate if absent)
    .github/workflows/ci.yml     GitHub Actions host gate (runs scripts/ci.sh)

  perf/ — LOCAL-ONLY, not pushed (.gitignore). Raw per-run measurement
  data is regenerated on the GPU host; current consolidated numbers
  live in PERFORMANCE.md. Historical campaign content is in
  archive/docs/ARCHIVE.md.

  Cross-project references:
    ~/MI300A_TARGET_ENVIRONMENT.md   definitive MI300A target reference
    ~/HIP_6900XT_KNOWLEDGE.md        definitive 6900 XT dev-side knowledge

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  9. C / HIP STYLE (loaded by .claude/rules/c-style.md)[0m

  See .claude/rules/c-style.md for the full C/HIP rules: language constraints,
  modular design, parameterization, hardware measurement, GPU kernel rules,
  modular arithmetic conventions, comment policy, and output/diagnostics.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m
