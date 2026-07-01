[1;37m╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          N T T   —   G P U   T D R   /   C R A S H   F I X   ( 6 9 0 0 X T )                           ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[0m

[1;33m  View with:  cat scripts/gpu_tdr_fix.md   or   less -R scripts/gpu_tdr_fix.md[0m

  Root cause of the GPU "system crashes" during long runs / full sweeps,
  and the environment change that lets the dev box tolerate sustained
  compute. The code-side mitigation (gpu_run.sh S18 timeout clamp) is
  already in place; THIS file is the host/driver fix that must be applied
  by the user (root + reboot) to enable genuinely long GPU sweeps.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  1. ROOT CAUSE  (diagnosed 2026-05-29)[0m

  The gfx1030 (RX 6900 XT) drives the DESKTOP and COMPUTE on one GPU
  (gdm3 / gnome-shell / Xwayland; single card1 + renderD128). The amdgpu
  hardware watchdog (lockup_timeout) is UNSET -> kernel default, which for
  a GPU-passthrough guest is 10000 ms (10 s) for ALL rings. gpu_recovery
  is -1 (auto). So any GPU job that occupies the GPU past ~10 s trips the
  watchdog; gpu_recovery then RESETS the GPU, which blackscreens the
  desktop = the "system crash".

  Confirming evidence:
   - /sys/module/amdgpu/parameters/lockup_timeout = (empty -> default)
   - /sys/module/amdgpu/parameters/gpu_recovery   = -1
   - perf/crash_diag/gpu_run_ledger.log: EVERY run < ~2 s exits rc=0;
     only the > 10 s monoliths (compute_e d=1e6 ~36 s; check-gpu suite)
     crashed.

  Implication: a wrapper timeout LONGER than 10 s cannot save the box —
  the hardware watchdog fires first. gpu_run.sh now clamps the per-run
  timeout to 8 s (S18) so runaways are aborted before the TDR. That keeps
  the box safe but also CAPS each run at 8 s of GPU time, which forbids
  long single-process sweeps. To lift that cap safely, apply the fix below.

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  2. THE FIX  (host, root + reboot)[0m

  Goal: let compute kernels run long WITHOUT the watchdog resetting the
  display GPU. Two configs, pick by risk tolerance. Create the file:

    sudo tee /etc/modprobe.d/amdgpu-tdr.conf

  OPTION A (recommended: long compute allowed, auto-recovery kept)
    options amdgpu lockup_timeout=10000,600000,600000,600000
    - Format [GFX,Compute,SDMA,Video] ms. GFX stays 10 s (display hang
      still detected); Compute/SDMA/Video raised to 10 min so long NTT
      compute is not killed. HIP kernels on RDNA2 dispatch to the COMPUTE
      queues, so this is the ring that matters for our sweeps.
    - If a kernel genuinely wedges, auto-recovery still fires (may
      blackscreen) — but legitimate long compute no longer trips it.

  OPTION B (never blackscreen; a real wedge needs a reboot)
    options amdgpu lockup_timeout=10000,600000,600000,600000 gpu_recovery=0
    - gpu_recovery=0 disables auto-reset entirely: a stuck job will hang
      that process but never reset the GPU / blackscreen the desktop.
      Recovery from a true wedge = reboot. Safest against crashes; least
      automatic.

  Apply:
    sudo update-initramfs -u        # if amdgpu is in the initramfs
    sudo reboot
  (Alternatively add the same `amdgpu.lockup_timeout=...` to the kernel
   cmdline in GRUB if amdgpu loads from initramfs before modprobe.d.)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  3. VERIFY  (after reboot)[0m

    cat /sys/module/amdgpu/parameters/lockup_timeout   # expect 10000,600000,...
    cat /sys/module/amdgpu/parameters/gpu_recovery      # 0 if Option B
  Then re-enable long sweeps in the wrapper by exporting:
    export GPU_RUN_ALLOW_LONG=1
  (only after verifying the fix; this lifts the S18 8 s clamp.)

[1;37m══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════[0m

[1;35m  4. VALIDATION PLAN  (escalate gently — do NOT jump to a full sweep)[0m

  Even with the fix, validate incrementally; passthrough+display+amdgpu
  is finicky and the fix is not guaranteed. With the user present:
   1. GPU_RUN_ALLOW_LONG=1 scripts/gpu_run.sh 30 compute_e_dev -d 100000   (~0.6 s)
   2. ... -d 300000   (watch: should now run ~2-3 s without reset)
   3. ... -d 1000000  (the original crasher, ~36 s) — ONLY once 1-2 pass.
   4. Then a full group_b_bench.sh sweep.
  Abort at the first reset and fall back to Option B (or keep the box as-is
  with the 8 s clamp and run sweeps as many short cells).

  STRUCTURAL ALTERNATIVE (best, if feasible): drive the desktop from a
  different adapter (iGPU / virtual display) so the gfx1030 is compute-only;
  display starvation then cannot trigger a GFX TDR at all.
