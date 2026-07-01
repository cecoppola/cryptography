#!/bin/bash
# mi300a_first_run.sh — one-button acceptance gate on the MI300A target.
#
# Runs after deploy: builds every gfx942 target from a clean tree, then
# steps through ROADMAP.md §4 acceptance criteria. Single pass / fail
# table and a single exit code. The MI300A operator runs this once;
# if it returns 0, Phase 4 is done.
#
# Layout: each block (B/M1..Mn) corresponds to one ROADMAP §4 checkbox.
# Cells that need 4 APUs auto-detect device count and skip with a clear
# message if they cannot be exercised (e.g. on a single-APU node).
#
# Output: a results/mi300a_first_run_<TS>.log with the full transcript
# plus a fixed-width summary table at the end. Bring the log home as
# the deploy-completion artifact.
#
# Usage:
#   bash scripts/mi300a_first_run.sh             # run all cells
#   bash scripts/mi300a_first_run.sh --build-only  # build proof only
#   COMPUTE_E_D=1000000 bash scripts/mi300a_first_run.sh
set -u
cd "$(dirname "$0")/.."

BUILD_ONLY=0
[ "${1:-}" = "--build-only" ] && BUILD_ONLY=1

COMPUTE_E_D="${COMPUTE_E_D:-1000000}"
LOG_N="${LOG_N:-20}"
N_TRIALS="${N_TRIALS:-100}"

mkdir -p results
LOG="results/mi300a_first_run_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

# First 200 fractional digits of e (OEIS A001113) — sanity ref for compute_e.
REF_E=71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901

declare -a NAMES RESULTS DETAIL
FAIL=0
add() {
    NAMES+=("$1"); RESULTS+=("$2"); DETAIL+=("$3")
    [ "$2" = "FAIL" ] && FAIL=1
    return 0
}

cell() {  # tag | runner-args... → captures rc + tail into LOG
    local tag="$1"; shift
    printf '  %-44s ... ' "$tag"
    {
        echo "===== $tag ====="
        echo "+ $*"
    } >>"$LOG"
    local out rc
    out=$("$@" 2>&1); rc=$?
    echo "$out" | tail -20 >>"$LOG"
    if [ "$rc" -eq 0 ]; then
        printf '\033[1;32mPASS\033[0m\n'
        add "$tag" "PASS" "$(echo "$out" | tail -1)"
    else
        printf '\033[1;31mFAIL (rc=%d)\033[0m\n' "$rc"
        add "$tag" "FAIL" "rc=$rc — see $LOG"
    fi
}

printf '\n\033[1;37m═══  MI300A first-run acceptance gate  ═══\033[0m\n\n'
printf '  log: %s\n' "$LOG"

# ── B. make all builds clean under PrgEnv-amd / ROCm 7.0.3 ───────────────────
printf '\n\033[1;36m── Build ──\033[0m\n'
{ make clean && make all-mi300a; } >>"$LOG" 2>&1 \
    && { printf '  %-44s ... \033[1;32mPASS\033[0m\n' 'B  make all-mi300a clean'; add 'B  make all-mi300a clean' PASS ''; } \
    || { printf '  %-44s ... \033[1;31mFAIL\033[0m\n' 'B  make all-mi300a clean'; add 'B  make all-mi300a clean' FAIL "see $LOG"; FAIL=1; }

if [ "$BUILD_ONLY" -eq 1 ]; then
    printf '\n  --build-only: stopping here (no runtime cells)\n'
    exit "$FAIL"
fi

# ── Device count detection ───────────────────────────────────────────────────
NDEV=$(rocminfo 2>/dev/null | grep -c 'gfx942' || true)
printf '\n  detected %d gfx942 device(s) on this node\n' "$NDEV" | tee -a "$LOG"
if [ "$NDEV" -lt 1 ]; then
    printf '  \033[1;31mno gfx942 found — this script must run on the MI300A node\033[0m\n'
    exit 1
fi

# ── M1. lib/test_ntt single-device correctness ──────────────────────────────
printf '\n\033[1;36m── M1  lib single-device correctness ──\033[0m\n'
( cd lib && make test_ntt ) >>"$LOG" 2>&1
cell 'M1  lib test_ntt (22/22 expected)'  srun -p mi -N 1 -n 1 ./lib/test_ntt

# ── M2. ref cross-verify (CPU vs gfx942 GPU element-wise) ───────────────────
printf '\n\033[1;36m── M2  ref CPU/GPU cross-verify ──\033[0m\n'
cell 'M2  ntt_cross_verify_mi300a ML-KEM'   srun -p mi -N 1 -n 1 ./bin/ntt_cross_verify_mi300a 256 3329 17

# ── M3. determinism on gfx942 ────────────────────────────────────────────────
printf '\n\033[1;36m── M3  gfx942 determinism (R1/R2/R3) ──\033[0m\n'
cell 'M3  determinism ML-KEM (50 reps)'     srun -p mi -N 1 -n 1 ./bin/ntt_gpu_determinism_mi300a 256 3329 17 50

# ── M4. G2  4-APU CRT polymul correctness ────────────────────────────────────
printf '\n\033[1;36m── M4  G2 4-APU CRT polymul (ROADMAP §4) ──\033[0m\n'
if [ "$NDEV" -ge 4 ]; then
    cell 'M4  G2 CRT polymul 4-APU log_n=20' \
        srun -p mi -N 1 -n 4 --ntasks-per-node=4 ./lib/crt_ntt "$LOG_N" "$N_TRIALS"
else
    printf '  %-44s ... \033[1;33mSKIP (need 4 APUs, have %d)\033[0m\n' 'M4  G2 CRT polymul 4-APU' "$NDEV"
    add 'M4  G2 CRT polymul 4-APU' SKIP "have $NDEV APU(s)"
fi

# ── M5. compute_e end-to-end vs OEIS ─────────────────────────────────────────
printf '\n\033[1;36m── M5  compute_e d=%s vs OEIS A001113 ──\033[0m\n' "$COMPUTE_E_D"
( cd app/compute_e && make all ) >>"$LOG" 2>&1
out=$(srun -p mi -N 1 -n "${NDEV}" --ntasks-per-node="${NDEV}" \
      ./app/compute_e/compute_e -d "$COMPUTE_E_D" 2>&1) ; rc=$?
echo "$out" | tail -3 >>"$LOG"
got=$(echo "$out" | grep -oE '^[0-9]+' | head -1 | head -c 200)
if [ "$rc" -eq 0 ] && [ "$got" = "$REF_E" ]; then
    printf '  %-44s ... \033[1;32mPASS\033[0m (first 200 digits match)\n' "M5  compute_e d=$COMPUTE_E_D OEIS-match"
    add "M5  compute_e d=$COMPUTE_E_D OEIS-match" PASS ''
else
    printf '  %-44s ... \033[1;31mFAIL\033[0m\n' "M5  compute_e d=$COMPUTE_E_D OEIS-match"
    add "M5  compute_e d=$COMPUTE_E_D OEIS-match" FAIL "see $LOG"
    FAIL=1
fi

# ── Summary table ────────────────────────────────────────────────────────────
printf '\n\033[1;37m──────────────────────────────────────────────────────────\033[0m\n'
printf '  \033[1;36mAcceptance summary\033[0m\n'
printf '\033[1;37m──────────────────────────────────────────────────────────\033[0m\n'
for i in "${!NAMES[@]}"; do
    case "${RESULTS[$i]}" in
        PASS) color='\033[1;32m';;
        FAIL) color='\033[1;31m';;
        SKIP) color='\033[1;33m';;
        *)    color='';;
    esac
    printf "  %-44s ${color}%-4s\033[0m\n" "${NAMES[$i]}" "${RESULTS[$i]}"
done | tee -a "$LOG"
printf '\n  log: %s\n' "$LOG"
if [ "$FAIL" -eq 0 ]; then
    printf '  \033[1;32mMI300A first-run: every cell PASS or SKIP — Phase 4 acceptance criteria met.\033[0m\n\n'
else
    printf '  \033[1;31mMI300A first-run: FAILED cell(s) — inspect %s before retrying.\033[0m\n\n' "$LOG"
fi
exit "$FAIL"
