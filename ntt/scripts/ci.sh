#!/bin/bash
# ci.sh — CI-agnostic entry point for the host reliability gate.
#
# Works the same whether invoked by GitHub Actions, GitLab CI, a local
# pre-commit/pre-push hook, or by hand. It (1) verifies the host
# toolchain is present, (2) runs `make check` (the full GPU-free gate),
# and (3) returns that single exit code.
#
# It does NOT build or run anything that needs ROCm/hipcc/a GPU — those
# suites stay gfx-gated and run only on the MI300A / 6900XT.
#
# Usage:   bash scripts/ci.sh
set -u
cd "$(dirname "$0")/.."

echo "── CI host gate ────────────────────────────────────────────"

# (1) Toolchain preflight — fail early with a clear message.
missing=0
for tool in cc gcc make gcov; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "  MISSING: $tool"; missing=1
    fi
done
# libgmp is needed by lib's GMP cross-oracle (test-arith-gmp).
if ! echo '#include <gmp.h>' | cc -E - >/dev/null 2>&1; then
    echo "  MISSING: gmp.h (install libgmp-dev / gmp-devel)"; missing=1
fi
if [ "$missing" -ne 0 ]; then
    echo "  toolchain preflight FAILED — install the items above."
    exit 1
fi
echo "  toolchain OK: cc gcc make gcov gmp"

# (2) Run the full host gate.
echo "  running: make check"
make check
rc=$?

# (3) Propagate the single exit code.
if [ "$rc" -eq 0 ]; then
    echo "── CI host gate: PASS ──────────────────────────────────────"
else
    echo "── CI host gate: FAIL (rc=$rc) ─────────────────────────────"
fi
exit "$rc"
