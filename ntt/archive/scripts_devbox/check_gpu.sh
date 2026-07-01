#!/bin/bash
# check_gpu.sh — unified GPU-side reliability gate for the 6900XT (gfx1030).
#
# Mirrors scripts/check.sh for the host suite. Runs every gfx1030 GPU
# correctness test we have, with one summary table and one exit code.
# Every GPU binary launched via scripts/gpu_run.sh (GPU-SAFETY rule).
#
# Cells (all under the watchdog cap; total runtime ~15-20 min):
#   R2  lib2 test_ntt_dev          (21/0 expected)
#   R3  lib2 test-modops           (4 primes)
#   R4  lib3 test_ntt_kernel       (6/6 KAT)
#   R5  lib3 test_polymul_sweep    (726/726 + 7/7 differential, LB=64)
#   R6  lib3 test_polymul_sweep    (726/726 + 7/7, LB=112)
#   R7  compute_e_dev -d 100       (100-digit smoke vs OEIS A001113)
#   R8  determinism, 11 primes     (R1/R2/R3 subtests per prime = 33 total)
#   R9  polymul binary U1 grid     (16/16, internal sweep on any invocation)
#   R10 stockham external sweep    (GPU-eligible primes × {256..16384})
#   R11 compute_e d=10000 dual-LB  (byte-identical LB=64 vs LB=112; F.7a)
#
# Build prerequisites: all GPU binaries from `make all` PLUS lib2/lib3
# in-place binaries. The driver builds them itself, fail-fast.
#
# Skipped: cross-verify sweep (R1) — covered by R2/R3/R4/R5/R6 in spirit
# and adds significant runtime; run `make check-gpu-full` to include it.
#
# Exit 0 only if every cell passes. Single source of truth for "is the
# 6900XT-reachable tree healthy?" — replaces the ad-hoc 2026-05-23
# GPU_SESSION_TASKS.md handoff.
set -u
cd "$(dirname "$0")/.."

# Allow the full variant via env or argv.
INCLUDE_R1="${INCLUDE_R1:-${CHECK_GPU_FULL:-0}}"
[ "${1:-}" = "--full" ] && INCLUDE_R1=1

mkdir -p results
LOG="results/check_gpu_$(date +%Y%m%d_%H%M%S).log"
: > "$LOG"

# ── GPU health gate ──────────────────────────────────────────────────────────
# Build prerequisites (make all) invoke hipcc, which opens the GPU device file.
# On a machine where a previous GPU crash left the amdgpu driver in a bad state,
# that open can re-crash the box (confirmed 2026-06-06). Abort before anything
# if the driver is unhealthy.
if scripts/gpu_health_check.sh 2>/dev/null; then
  :  # healthy or no GPU node — proceed
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    printf '\033[1;31m[check-gpu] GPU driver appears unhealthy — power cycle required.\033[0m\n' >&2
    printf '  A previous GPU crash may have corrupted the amdgpu driver state.\n' >&2
    printf '  Power cycle (not just reboot) before re-running make check-gpu.\n' >&2
    exit 1
  fi
fi

# Reference: first 200 fractional digits of e (OEIS A001113) for R7 + R11.
REF_E=71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901

# ── Build prerequisites ──────────────────────────────────────────────────────
printf '\n\033[1;37m═══  make check-gpu — 6900XT reliability gate  ═══\033[0m\n\n'
printf '  building prerequisites (root all + lib2 dev/test_ntt/arith + lib3 dev/f7) ... '
{
    make all >>"$LOG" 2>&1 \
        && make -C lib2 dev test_ntt arith test_modops >>"$LOG" 2>&1 \
        && make -C lib3/compute_e dev f7-build >>"$LOG" 2>&1 \
        && make -C lib3/compute_e test_kernel_dev test_polymul_sweep f7-polymul112 >>"$LOG" 2>&1
} || { printf '\033[1;31mBUILD FAILED — see %s\033[0m\n' "$LOG"; exit 1; }
printf '\033[1;32mok\033[0m\n\n'

# ── Run cells ────────────────────────────────────────────────────────────────
declare -a NAMES RESULTS DETAIL
FAIL=0
add() { NAMES+=("$1"); RESULTS+=("$2"); DETAIL+=("$3"); [ "$2" = "FAIL" ] && FAIL=1; }

cell() {  # name|wrap-secs|binary args... → captures rc + last 20 lines into LOG, returns
    local name="$1" secs="$2"; shift 2
    printf '  running %-32s ... ' "$name"
    {
        echo "===== $name ====="
        echo "+ scripts/gpu_run.sh $secs $*"
    } >>"$LOG"
    local out rc
    out=$(scripts/gpu_run.sh "$secs" "$@" 2>&1); rc=$?
    # rc=99 = wrapper safety STOP (post-check "GPU still busy", S15, etc.) —
    # NOT a test failure. Retry once after an extended cooldown; the second
    # attempt almost always succeeds because the transient has cleared.
    if [ "$rc" -eq 99 ]; then
        echo "[check_gpu] rc=99 transient — cooling 30s + retry" >>"$LOG"
        sleep 30
        out=$(scripts/gpu_run.sh "$secs" "$@" 2>&1); rc=$?
    fi
    echo "$out" >>"$LOG"
    if [ "$rc" -eq 0 ]; then
        printf '\033[1;32mPASS\033[0m '
    else
        printf '\033[1;31mFAIL\033[0m (rc=%s) ' "$rc"
    fi
    LAST_OUT="$out"; LAST_RC=$rc
}

# R2 — lib2 test_ntt_dev
cell "R2 lib2 test_ntt_dev"        180 lib2/test_ntt_dev
p=$(echo "$LAST_OUT" | grep -c '\[PASS\]'); f=$(echo "$LAST_OUT" | grep -c '\[FAIL\]')
if [ "$LAST_RC" -eq 0 ] && [ "$f" -eq 0 ] && [ "$p" -gt 0 ]; then
    add "R2 lib2 test_ntt_dev"        PASS "$p/$p"; printf '%s/%s\n' "$p" "$p"
else add "R2 lib2 test_ntt_dev"      FAIL "rc=$LAST_RC p=$p f=$f"; printf '\n'; fi

# R3 — lib2 test_modops
cell "R3 lib2 test_modops"         300 ./lib2/test_modops
p=$(echo "$LAST_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -cE 'P[0-9].*PASS'); f=$(echo "$LAST_OUT" | grep -c 'FAIL')
if [ "$LAST_RC" -eq 0 ] && [ "$f" -eq 0 ] && [ "$p" -ge 4 ]; then
    add "R3 lib2 test_modops"          PASS "$p/4 primes"; printf '%s/4 primes\n' "$p"
else add "R3 lib2 test_modops"         FAIL "rc=$LAST_RC p=$p f=$f"; printf '\n'; fi

# R4 — lib3 test_ntt_kernel KA1-KA6
cell "R4 lib3 test_ntt_kernel"     120 lib3/compute_e/test_kernel_dev
m=$(echo "$LAST_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/[0-9]+ passed' | tail -1)
if [ "$LAST_RC" -eq 0 ] && [ -n "$m" ]; then
    add "R4 lib3 test_ntt_kernel"     PASS "$m"; printf '%s\n' "$m"
else add "R4 lib3 test_ntt_kernel"    FAIL "rc=$LAST_RC m=$m"; printf '\n'; fi

# R5 — lib3 test_polymul_sweep LB=64
cell "R5 polymul_sweep LB=64"      600 lib3/compute_e/test_polymul_sweep 15
m=$(echo "$LAST_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'diff [0-9]+/[0-9]+, large [0-9]+/[0-9]+' | tail -1)
if [ "$LAST_RC" -eq 0 ] && echo "$LAST_OUT" | grep -q 'POLYMUL SWEEP PASS'; then
    add "R5 polymul_sweep LB=64"      PASS "$m"; printf '%s\n' "$m"
else add "R5 polymul_sweep LB=64"     FAIL "rc=$LAST_RC"; printf '\n'; fi

# R6 — lib3 test_polymul_sweep LB=112
cell "R6 polymul_sweep LB=112"     600 lib3/compute_e/test_polymul_sweep_l112 15
m=$(echo "$LAST_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE 'diff [0-9]+/[0-9]+, large [0-9]+/[0-9]+' | tail -1)
if [ "$LAST_RC" -eq 0 ] && echo "$LAST_OUT" | grep -q 'POLYMUL SWEEP PASS'; then
    add "R6 polymul_sweep LB=112"     PASS "$m"; printf '%s\n' "$m"
else add "R6 polymul_sweep LB=112"    FAIL "rc=$LAST_RC"; printf '\n'; fi

# R7 — compute_e d=100 smoke (host vs OEIS)
printf '  running %-32s ... ' "R7 compute_e d=100 vs OEIS"
out=$(lib3/compute_e/compute_e_dev -d 100 2>&1); rc=$?
{ echo "===== R7 compute_e_dev -d 100 ====="; echo "$out"; } >>"$LOG"
val=$(echo "$out" | sed -nE 's/^e = 2\.([0-9]+).*/\1/p')
gpu_calls=$(echo "$out" | grep -oE 'gpu_dispatches: [0-9]+' | grep -oE '[0-9]+$' || echo 0)
want=${REF_E:0:${#val}}
# d=100 uses schoolbook (all operands < 64-limb GPU threshold); gpu_calls may be 0.
# R5/R6 test_polymul_sweep verify GPU NTT pipeline; R7 verifies output correctness.
if [ "$rc" -eq 0 ] && [ -n "$val" ] && [ "$val" = "$want" ]; then
    add "R7 compute_e d=100"           PASS "${#val} digits match (gpu_dispatches=$gpu_calls)"; printf '\033[1;32mPASS\033[0m %d digits (gpu_dispatches=%s)\n' "${#val}" "$gpu_calls"
else add "R7 compute_e d=100"          FAIL "rc=$rc"; printf '\033[1;31mFAIL\033[0m (rc=%s)\n' "$rc"; fi

# R8 — determinism, all 11 GPU-eligible primes at n=256 (3 sub-tests each = 33 total)
# omegas computed as g^((q-1)/256) mod q; see lib1/ntt_moduli.h for g values
R8_PASS=0; R8_TOTAL=0
for cfg in \
    "256 257 3"           \
    "256 3329 3061"       \
    "256 7681 2028"       \
    "256 12289 8340"      \
    "256 40961 36043"     \
    "256 65537 282"       \
    "256 8380417 6644104" \
    "256 167772161 26238939"  \
    "256 469762049 338628632" \
    "256 998244353 476477967" \
    "256 2013265921 1732600167"; do
    cell "R8 det $cfg"               120 bin/ntt_gpu_determinism_6900xt $cfg 50
    pp=$(echo "$LAST_OUT" | grep -c '1;32mPASS'); ff=$(echo "$LAST_OUT" | grep -c '1;31mFAIL')
    R8_TOTAL=$((R8_TOTAL+pp+ff))
    if [ "$LAST_RC" -eq 0 ] && [ "$ff" -eq 0 ] && [ "$pp" -ge 3 ]; then
        R8_PASS=$((R8_PASS+pp)); printf '%d/3 sub\n' "$pp"
    else printf '\n'; fi
done
if [ "$R8_PASS" -ge 33 ]; then add "R8 determinism (11 primes)" PASS "$R8_PASS/33 sub"; else add "R8 determinism (11 primes)" FAIL "$R8_PASS/33 sub"; fi

# R9 — polymul binary internal U1 sweep (one invocation = 16/16)
cell "R9 U1 negacyclic (polymul U1 grid)" 180 bin/ntt_gpu_polymul_6900xt 256 3329 3061 10
m=$(echo "$LAST_OUT" | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/[0-9]+ passed' | tail -1)
if [ "$LAST_RC" -eq 0 ] && [ "$m" = "16/16 passed" ]; then
    add "R9 polymul U1 grid"          PASS "$m"; printf '%s\n' "$m"
else add "R9 polymul U1 grid"         FAIL "rc=$LAST_RC m=$m"; printf '\n'; fi

# R10 — external Stockham sweep over the 14-prime GPU-eligible grid
python3 << 'PYEOF' > /tmp/check_gpu_cells.txt
def mp(b,e,m):
    r=1; b%=m
    while e: r=r*b%m if e&1 else r; b=b*b%m; e>>=1
    return r
TABLE=[("Fermat-8",257,3,8),("ML-KEM",3329,3,8),("ML-KEM-v0",7681,17,9),
       ("FALCON",12289,11,12),("Proth-5-13",40961,3,13),("Fermat-16",65537,3,16),
       ("ML-DSA",8380417,10,13),("CRT-lo",167772161,3,25),("CRT-mid",469762049,3,26),
       ("CRT-hi",998244353,3,23),("FHE-RNS",2013265921,31,27)]
for nm,q,g,ml in TABLE:
    nmax=1<<ml
    for n in (256,512,1024,2048,4096,16384):
        if n>nmax: continue
        print(f'{n} {q} {mp(g,(q-1)//n,q)}')
PYEOF
R10_T=0; R10_P=0
while read -r n q w; do
    R10_T=$((R10_T+1))
    out=$(scripts/gpu_run.sh 60 bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" 1 2>&1); rc=$?
    # rc=99 retry-once (same rationale as cell()): wrapper transient.
    if [ "$rc" -eq 99 ]; then sleep 30; out=$(scripts/gpu_run.sh 60 bin/ntt_gpu_stockham_6900xt "$n" "$q" "$w" 1 2>&1); rc=$?; fi
    if [ "$rc" -eq 0 ] && ! echo "$out" | sed 's/\x1b\[[0-9;]*m//g' | grep -q "FAIL"; then
        R10_P=$((R10_P+1))
    else
        echo "===== R10 stockham FAIL n=$n q=$q w=$w rc=$rc =====" >>"$LOG"
        echo "$out" >>"$LOG"
    fi
done < /tmp/check_gpu_cells.txt
printf '  running %-32s ... ' "R10 stockham sweep ($R10_T cells)"
if [ "$R10_P" = "$R10_T" ] && [ "$R10_T" -gt 0 ]; then
    add "R10 stockham sweep ($R10_T cells)" PASS "$R10_P/$R10_T"; printf '\033[1;32mPASS\033[0m %d/%d\n' "$R10_P" "$R10_T"
else add "R10 stockham sweep"      FAIL "$R10_P/$R10_T"; printf '\033[1;31mFAIL\033[0m %d/%d\n' "$R10_P" "$R10_T"; fi

# R11 — compute_e d=10000 LB=64 vs LB=112 byte-identical (F.7a)
printf '  running %-32s ... ' "R11 compute_e d=10000 dual-LB"
o64=$(scripts/gpu_run.sh 600 lib3/compute_e/compute_e_dev_l64 -d 10000 2>&1); rc64=$?
o112=$(scripts/gpu_run.sh 600 lib3/compute_e/compute_e_dev_l112 -d 10000 2>&1); rc112=$?
{ echo "===== R11 LB=64 ====="; echo "$o64"; echo "===== R11 LB=112 ====="; echo "$o112"; } >>"$LOG"
v64=$(echo "$o64" | grep '^e = ' | head -1); v112=$(echo "$o112" | grep '^e = ' | head -1)
if [ "$rc64" -eq 0 ] && [ "$rc112" -eq 0 ] && [ -n "$v64" ] && [ "$v64" = "$v112" ]; then
    add "R11 compute_e d=10000 dual"   PASS "byte-identical (${#v64}B)"; printf '\033[1;32mPASS\033[0m byte-identical %dB\n' "${#v64}"
else add "R11 compute_e d=10000 dual" FAIL "rc=$rc64/$rc112 match=$([ "$v64" = "$v112" ] && echo yes || echo no)"; printf '\033[1;31mFAIL\033[0m\n'; fi

# R1 (optional, --full) — cross-verify sweep over GPU-eligible primes
if [ "$INCLUDE_R1" = "1" ]; then
    python3 << 'PYEOF' > /tmp/check_gpu_r1_cells.txt
def mp(b,e,m):
    r=1; b%=m
    while e: r=r*b%m if e&1 else r; b=b*b%m; e>>=1
    return r
TABLE=[("Fermat-8",257,3,8),("ML-KEM",3329,3,8),("ML-KEM-v0",7681,17,9),
       ("FALCON",12289,11,12),("Proth-5-13",40961,3,13),("Fermat-16",65537,3,16),
       ("ML-DSA",8380417,10,13),("CRT-lo",167772161,3,25),("CRT-mid",469762049,3,26),
       ("CRT-hi",998244353,3,23),("FHE-RNS",2013265921,31,27)]
for nm,q,g,ml in TABLE:
    nmax=1<<ml
    for n in (256,512,1024,2048,4096,16384):
        if n>nmax: continue
        print(f'{n} {q} {mp(g,(q-1)//n,q)}')
PYEOF
    R1_T=0; R1_P=0; R1_SUB_P=0; R1_SUB_T=0
    while read -r n q w; do
        R1_T=$((R1_T+1))
        out=$(scripts/gpu_run.sh 60 bin/ntt_cross_verify_6900xt "$n" "$q" "$w" 2>&1); rc=$?
        if [ "$rc" -eq 99 ]; then sleep 30; out=$(scripts/gpu_run.sh 60 bin/ntt_cross_verify_6900xt "$n" "$q" "$w" 2>&1); rc=$?; fi
        res=$(echo "$out" | sed 's/\x1b\[[0-9;]*m//g' | grep "Results:" | tail -1)
        p=$(echo "$res" | sed -nE 's/.*Results: *([0-9]+) passed +([0-9]+) failed.*/\1/p'); p=${p:-0}
        f=$(echo "$res" | sed -nE 's/.*Results: *([0-9]+) passed +([0-9]+) failed.*/\2/p'); f=${f:-0}
        R1_SUB_T=$((R1_SUB_T+p+f)); R1_SUB_P=$((R1_SUB_P+p))
        [ "$rc" = "0" ] && [ "$f" = "0" ] && [ "$p" -gt 0 ] && R1_P=$((R1_P+1))
    done < /tmp/check_gpu_r1_cells.txt
    printf '  running %-32s ... ' "R1 cross-verify sweep ($R1_T cells)"
    if [ "$R1_P" = "$R1_T" ]; then
        add "R1 cross-verify sweep ($R1_T cells)" PASS "$R1_SUB_P/$R1_SUB_T subtests"; printf '\033[1;32mPASS\033[0m %d/%d cells, %d/%d sub\n' "$R1_P" "$R1_T" "$R1_SUB_P" "$R1_SUB_T"
    else add "R1 cross-verify sweep" FAIL "$R1_P/$R1_T"; printf '\033[1;31mFAIL\033[0m\n'; fi
fi

# ── Summary table ───────────────────────────────────────────────────────────
printf '\n\033[1;37m┌────────────────────────────────────────────┬────────┬────────────────────────────┐\033[0m\n'
printf '\033[1;37m│\033[0m \033[1;36m%-42s\033[0m \033[1;37m│\033[0m \033[1;36m%-6s\033[0m \033[1;37m│\033[0m \033[1;36m%-26s\033[0m \033[1;37m│\033[0m\n' "Cell" "Result" "Detail"
printf '\033[1;37m├────────────────────────────────────────────┼────────┼────────────────────────────┤\033[0m\n'
for i in "${!NAMES[@]}"; do
    if [ "${RESULTS[$i]}" = "PASS" ]; then c='\033[1;32m'; else c='\033[1;31m'; fi
    printf '\033[1;37m│\033[0m %-42s \033[1;37m│\033[0m '"$c"'%-6s\033[0m \033[1;37m│\033[0m %-26s \033[1;37m│\033[0m\n' \
        "${NAMES[$i]}" "${RESULTS[$i]}" "${DETAIL[$i]}"
done
printf '\033[1;37m└────────────────────────────────────────────┴────────┴────────────────────────────┘\033[0m\n'

if [ "$FAIL" -eq 0 ]; then
    printf '\n  \033[1;32mGPU gate: all %d cells PASS\033[0m   log=%s\n\n' "${#NAMES[@]}" "$LOG"
else
    printf '\n  \033[1;31mGPU gate: FAILURES present — see %s\033[0m\n\n' "$LOG"
fi
exit "$FAIL"
