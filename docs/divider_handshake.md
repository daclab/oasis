# Calyx/HardFloat divider handshake repair

## Failure and evidence

On 2026-10-21, the v0 small attention benchmark completed in 9,576,728 cycles
but failed the PyTorch comparison: 509 of 512 outputs mismatched, with maximum
absolute error 0.053414054214954376. The saved Verilator 5.052 run and the user's
subsequent Vivado xsim run produced the same reported failure. This is an RTL
correctness problem, not a simulator timeout. The failed run remains in
`out/attention_small/`.

Inspection of the saved intermediate memories showed query/key/value projections
and raw scores agreeing with the CPU reference to about 5e-7. Errors appeared in
the scaling/softmax section. An isolated Verilator 5.052 divider experiment on
2026-10-21 reproduced 20 failures in 24 requests. For example, `3 / 2` returned
`0.5`, the result of the preceding `1 / 2`. An isolated candidate using the same
HardFloat arithmetic passed all 24 bit-exact checks. These results establish a
wrapper defect; they do not establish that it was the only attention defect.

## Cause

The pinned Calyx `primitives/float/divSqrtFN.sv` wrapper connects Calyx `go`
directly to HardFloat `inValid`. Calyx holds go while waiting for completion.
HardFloat can become ready again while go remains high, accepting the same request
twice. The trace recorded acceptance of the first request at cycles 4 and 29.

Independently, the wrapper shifts go through a 32-stage `done_buf` and exposes its
last bit as done. A sustained go creates a sustained, delayed done. Later caller
transactions therefore see a completion belonging to an earlier operation and
consume stale data. Although the original wrapper latches its output on outValid,
its completion signal is unrelated to that event. Increasing the fixed delay
would not repair either defect.

HardFloat accepts a request when inValid and inReady are high together at a rising
clock edge. It announces the valid result with outValid; division/square-root
latency is not guaranteed constant. See the [official HardFloat module contract,
section 9.8](https://www.jhauser.us/arithmetic/HardFloat-1/doc/HardFloat-Verilog.html#divSqrtRecFN_small).
The arithmetic core is retained unchanged.

The Calyx primitive declaration also carries `@interval(32)`. The pinned Calyx
compiler's static inference/promotion passes consume this annotation. Leaving it
in place after introducing variable completion would retain a contradictory
scheduling contract, so the generated declaration must be dynamic as well.

## Permanent solution

- `dynamic_divsqrt_import` in `python/oasis/backend/fixups.py` replaces the divider
  import with its extern declaration without `@interval(32)`, before Calyx
  optimization. The external RTL path comes from configured `calyx_lib`; the
  installed Calyx checkout is not edited.
- `repair_divsqrt_handshake` replaces the known faulty wrapper with the packaged
  `backend/templates/divsqrt_handshake.sv`. Code hashes reject unexpected upstream
  changes instead of silently applying a stale repair. Reapplying is idempotent.
- The wrapper presents `request_valid = go && !busy && !done && !reset`, independent
  of inReady. Acceptance sets busy and retains the rounding control. While busy,
  no new request is presented. On outValid, it latches the rounded result and
  exception flags, clears busy, and raises done. Done is held until go is released.
  Reset clears busy, completion, result, and flags and resets HardFloat.
- `prepare_rtl` is shared by the final simulation stage and
  `generate_synth_sv`, so both Vivado and Yosys synthesis RTL receive the same
  functional repair. Existing port-redeclaration cleanup is retained.
- The pipeline still uses the pinned tools and the existing frontend/backend
  text-file boundary. No external dependency version was changed.

An additional CLI correction retains `--out` in the printed check/report commands.
Without it, checking a separately generated fixed design could accidentally read
the earlier failed default directory.

## Validation and rerun

`tests/test_divsqrt.py` checks unknown-wrapper rejection, idempotence, and identical
repairs through simulation/synthesis generation entry points. Its short real RTL
test uses configured Calyx and Verilator, runs against both original and repaired
wrappers, and checks consecutive divisions, square roots, zero/sign handling,
infinity, divide-by-zero and inexact flags, held completion, reset during a
transaction, and Calyx-generated control for successive divisions. It never runs
Vivado or synthesis. Logs from the full test invocation are under
`out/divider_fix_tests/test_real_divider_handshake0/`.

The fixed v0 attention RTL, testbench, and synthesis scripts have been generated
in `out/attention_small_fixed/`, preserving the failed run. The user completed the repaired v0 small attention simulation on 2026-10-21
with Verilator 5.052: PASS, 0/512 mismatches, maximum absolute error
3.501772880554199e-07, 9,603,371 cycles, no timeout. The agent re-ran
`oasis check attention --size small --out out/attention_small_fixed` against
the saved output and confirmed PASS without rerunning simulation. Medium
attention RTL, repaired xsim, and synthesis remain unverified. Under AGENTS.md, full/long EDA
runs are left to the user; generating synthesis RTL is not a synthesis run.

Run the already generated fixed design:

```bash
conda activate oasis
bash out/attention_small_fixed/sim/run_verilator.sh
oasis check attention --size small --out out/attention_small_fixed
```

For xsim instead, use `bash out/attention_small_fixed/sim/run_xsim.sh`, followed
by the same check command. Do not run the scripts in the old output directory
and expect the new repair. Do not increase comparison tolerances to mask errors.

To regenerate explicitly:

```bash
oasis compile attention --size small --out out/attention_small_fixed
```

A successful primitive regression is not a full attention PASS. If the repaired
attention run still fails, compare intermediate memories again to locate the
next defect, preserving the divider regression and existing tolerances.
