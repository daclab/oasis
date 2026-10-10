"""OASIS command line.

oasis compile gemm      PyTorch -> MLIR -> Calyx -> NN_design.sv (+ inputs.npz, golden.npy),
                        every stage's IR, then the testbench (same as `oasis tb`)
                        --no-dump-all: keep only the files later stages need; --no-tb: skip tb
oasis tb gemm           sim/ (tb.sv, mem_N.dat, run_xsim.sh) and synth/ (RTL + scripts)
oasis sim gemm          run sim/run_xsim.sh (Vivado xsim), then check
oasis check gemm        compare the simulated output memory with golden.npy
oasis synth gemm        run synth/run_synth.sh (Vivado) or run_yosys.sh ([fpga] synth_tool),
                        then parse the reports
oasis report gemm       re-read logs/reports and print the summary (no tools run)
oasis compile gemm --sim --synth
                        ...then also simulate (+ check) and/or synthesize (+ report)
oasis run gemm          same as `oasis compile gemm --sim --synth`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from oasis import __version__, report
from oasis.backend import memmap, testbench, verilator, vivado, yosys
from oasis.backend.fixups import FIXUPS
from oasis.backend.memmap import MemMapError
from oasis.config import ToolNotFoundError, load_pipeline, load_toolchain
from oasis.data import read_out
from oasis.frontend.export import export_linalg
from oasis.frontend.lower import run_frontend_passes
from oasis.frontend.prepare import with_weight_args
from oasis.inputs import make_inputs, save_inputs
from oasis.models import load_benchmark, registry
from oasis.stages import StageError, Workspace, check_handoff, run_tool
from oasis.verify import compare, golden, save_golden


def _toolchain(args: argparse.Namespace):
    return load_toolchain(synth_tool=getattr(args, "synth_tool", None))


def _workspace(args: argparse.Namespace, dump_all: bool = False):
    """(benchmark, workspace) for the benchmark/size/out arguments shared by all commands."""
    bench = load_benchmark(args.benchmark, args.size)
    out = Path(args.out) if args.out else Path("out") / f"{bench.name}_{bench.size}"
    return bench, Workspace(out, dump_all=dump_all)


def cmd_compile(args: argparse.Namespace) -> int:
    """Run every stage of the pipeline on a benchmark from models/data.py."""
    if (args.sim or args.synth) and (not args.tb or args.stop_after):
        raise StageError(
            "compile",
            "--sim/--synth need the full compile and the testbench (drop --no-tb / --stop-after)",
        )
    toolchain = _toolchain(args)
    pipeline = load_pipeline(Path(args.pipeline) if args.pipeline else None)
    bench, ws = _workspace(args, dump_all=args.dump_all)

    if bench.status == "wip":
        print(f"note: {bench.name} is work in progress: {bench.note}", flush=True)
    # The model as compiled (seeded, prepared), its inputs and the PyTorch reference output.
    model = bench.build()
    activations = make_inputs(bench)
    if bench.weights == "args":
        export_model, export_args = with_weight_args(model, activations)
    else:
        export_model, export_args = model, activations
    save_inputs(export_args, ws.out)  # arg0..argN: every kernel input, weights included
    save_golden(golden(model, activations), ws.out)
    report.reset(ws.out)
    report.update(ws.out, "compile", {"top": args.top, "dtype": bench.dtype})

    stale = ws.clear_dumps()
    if stale:
        print(f"removed {len(stale)} stage output(s) of an earlier compile")

    text = ""
    prev_path: Path | None = None  # previous stage's output on disk (input of tool stages)
    for index, stage in enumerate(pipeline.stages):
        log = ws.log_path(stage.name)
        print(f"[{index:02d}] {stage.name:<12} ({stage.kind})", flush=True)
        if stage.kind == "export":
            text, graphs = export_linalg(export_model, export_args, func_name=args.top)
            log.write_text(
                "torch_mlir.fx.export_and_import(output_type='linalg-on-tensors')\n\n" + graphs
            )
        elif stage.kind == "frontend":
            text = run_frontend_passes(text, stage.passes, stage.name, toolchain, log)
        elif stage.kind == "tool":
            if prev_path is None:
                raise StageError(stage.name, "previous stage's output is not on disk")
            out_path = ws.dump_path(index, stage.name, stage.ext)
            subs = {"in": prev_path, "out": out_path, "top": args.top}
            subs |= {
                k: toolchain.path(k)
                for k in toolchain.paths
                if "{" + k + "}" in " ".join(stage.args)
            }
            tool_args = [a.format(**subs) for a in stage.args]
            run_tool(
                stage.name,
                toolchain.resolve(stage.tool),
                tool_args,
                out_path,
                log,
                stdout_to_file=stage.stdout,
                unlimited_stack=stage.unlimited_stack,
            )
            prev_path = out_path
        elif stage.kind == "fixup":
            if prev_path is None:
                raise StageError(stage.name, "previous stage's output is not on disk")
            if stage.fixup not in FIXUPS:
                raise StageError(stage.name, f"unknown fixup '{stage.fixup}'")
            out_path = ws.dump_path(index, stage.name, stage.ext)
            FIXUPS[stage.fixup](prev_path, out_path, ws, log)
            prev_path = out_path
        else:
            raise StageError(stage.name, f"unknown stage kind '{stage.kind}'")

        if stage.kind not in ("tool", "fixup"):
            written = ws.write_dump(index, stage.name, text, force=stage.handoff)
            prev_path = written
            if stage.handoff:
                check_handoff(text, pipeline.handoff_dialects)
                print(f"     hand-off IR OK -> {written}")
        if args.stop_after == stage.name:
            print(f"stopped after '{stage.name}': outputs in {ws.out}")
            return 0

    print(f"done: outputs in {ws.out}")
    if not args.tb:
        return 0
    print("testbench:")
    rc = cmd_tb(args)
    if rc:
        return rc
    # Vivado runs, only when asked for: they are slow.
    if args.sim:
        print("simulation (Vivado xsim):")
        rc = cmd_sim(args)  # 2 = simulation ran but the check failed; still run synthesis
    if args.synth:
        print(f"synthesis ({_toolchain(args).fpga.synth_tool}):")
        rc = cmd_synth(args) or rc
    return rc


def cmd_tb(args: argparse.Namespace) -> int:
    """Write sim/ (testbench, memory files, one script per installed simulator) and synth/."""
    tc = _toolchain(args)
    bench, ws = _workspace(args)
    top = report.load(ws.out).get("compile", {}).get("top", "forward")

    inputs_npz = np.load(ws.out / "inputs.npz")
    inputs = [inputs_npz[f"arg{i}"] for i in range(len(inputs_npz.files))]
    expected = np.load(ws.out / "golden.npy")
    memories = memmap.bind(memmap.parse_memories(ws.find("calyx").read_text()), inputs, expected)

    sim_dir = ws.out / "sim"
    testbench.write_sim_dir(
        sim_dir, memories, inputs, bench.dtype, tc.fpga.clock_mhz, tc.cycle_limit
    )
    # One run script per simulator that is installed; `--simulator` / [sim] picks which runs.
    design = ws.find("design")
    sim_scripts = {}
    if all(tc.available(k) for k in ("xvlog", "xelab", "xsim")):
        sim_scripts["xsim"] = vivado.write_xsim_script(sim_dir, design, tc)
    if tc.available("verilator"):
        sim_scripts["verilator"] = verilator.write_verilator_script(sim_dir, design, tc)

    synth_script = None
    tool = tc.fpga.synth_tool
    if tc.available(tool):
        synth_dir = ws.out / "synth"
        synth_dir.mkdir(exist_ok=True)
        # Scripts and results of either tool from an earlier `oasis tb` belong to old RTL.
        for name in (*vivado.SYNTH_FILES, *yosys.SYNTH_FILES):
            (synth_dir / name).unlink(missing_ok=True)
        synth_sv = synth_dir / "design_synth.sv"
        vivado.generate_synth_sv(
            ws.find("futil"), synth_sv, tc, synth_dir / "calyx.log", disable_verify=tool == "yosys"
        )
        synth_top = tc.fpga.synth_top.format(top=top)
        backend = yosys if tool == "yosys" else vivado
        synth_script = backend.write_synth_scripts(synth_dir, synth_sv, synth_top, tc)

    for mem in memories:
        print(f"  {mem.name}: arg {mem.arg_index} ({mem.role}, {mem.shape}, {mem.width}-bit)")
    print(f"testbench: {sim_dir}/tb.sv   ({tc.fpga.clock_mhz:g} MHz clock)")
    check = f"oasis check {bench.name} --size {bench.size}"
    for name, script in sim_scripts.items():
        print(f"simulate ({name}): bash {script}    then: {check}")
    if not sim_scripts:
        print("simulate:  no simulator found (xsim or verilator); see `oasis tools`")
    if synth_script:
        print(
            f"synthesis ({tool}): bash {synth_script}    "
            f"then: oasis report {bench.name} --size {bench.size} --synth-tool {tool}"
        )
        target = tc.fpga.yosys_family if tool == "yosys" else tc.fpga.part
        print(f"           top = {synth_top}, target = {target}")
    else:
        print(f"synthesis: skipped ({tool} not found; see `oasis tools`)")
    return 0


def _collect_sim(ws: Workspace) -> dict:
    """Cycle count from sim/sim.log -> report.json (replacing any earlier sim result)."""
    log = ws.out / "sim" / "sim.log"
    if not log.exists():
        raise StageError("sim", f"{log} not found; run a sim/run_*.sh script (or `oasis sim`)")
    return report.update(ws.out, "sim", vivado.parse_sim_log(log.read_text()))


def _check(bench, ws: Workspace) -> bool:
    """Compare the output memory dump with golden.npy -> report.json."""
    sim_dir = ws.out / "sim"
    rep = _collect_sim(ws)
    if rep["sim"].get("timeout"):
        raise StageError("check", "simulation hit the cycle limit; see sim/sim.log")
    if rep["sim"].get("cycles") is None:
        raise StageError(
            "check", "simulation has not finished yet (no 'Simulated N cycles' in sim/sim.log)"
        )
    (out_mem,) = [m for m in memmap.load(sim_dir / "memmap.json") if m.role == "output"]
    dump = sim_dir / f"{out_mem.name}.out"
    if not dump.exists():
        raise StageError("check", f"{dump} not found; did the simulation finish? see sim/sim.log")
    actual = read_out(dump, out_mem.shape, bench.dtype, out_mem.width)
    np.save(sim_dir / "sim_out.npy", actual)
    result = compare(actual, np.load(ws.out / "golden.npy"), bench.dtype)
    report.update(
        ws.out,
        "check",
        {"ok": result.ok, "max_abs_err": result.max_abs_err, "mismatches": result.mismatches},
    )
    return result.ok


def cmd_sim(args: argparse.Namespace) -> int:
    bench, ws = _workspace(args)
    simulator = getattr(args, "simulator", None) or _toolchain(args).simulator
    script = ws.out / "sim" / f"run_{simulator}.sh"
    if not script.exists():
        raise StageError(
            "sim", f"{script} not found: is {simulator} installed? (`oasis tools`, then `oasis tb`)"
        )
    vivado.run_script(script, "sim")
    ok = _check(bench, ws)
    print(report.summary(report.load(ws.out)))
    return 0 if ok else 2


def cmd_check(args: argparse.Namespace) -> int:
    bench, ws = _workspace(args)
    ok = _check(bench, ws)
    print(report.summary(report.load(ws.out)))
    return 0 if ok else 2


_SYNTH_SCRIPT = {"vivado": "run_synth.sh", "yosys": "run_yosys.sh"}
_SYNTH_RESULT = {"vivado": "utilization.rpt", "yosys": "stat.json"}


def _collect_synth(ws: Workspace, args: argparse.Namespace) -> None:
    """Synthesis results of the configured tool -> report.json (replacing earlier ones)."""
    tc = _toolchain(args)
    top = report.load(ws.out).get("compile", {}).get("top", "forward")
    if tc.fpga.synth_tool == "yosys":
        stat = ws.out / "synth" / "stat.json"
        if not stat.exists():
            raise StageError("synth", f"{stat} not found; run synth/run_yosys.sh first")
        values = yosys.parse_stat(stat.read_text())
        values |= {
            "tool": "yosys",
            "top": tc.fpga.synth_top.format(top=top),
            "family": tc.fpga.yosys_family,
        }
        netlist = ws.out / "synth" / "netlist.v"
        if netlist.exists():
            values["netlist"] = str(netlist.relative_to(ws.out))
        report.replace(ws.out, "synth", values)
        return
    synth_dir = ws.out / "synth"
    util_rpt, timing_rpt = synth_dir / "utilization.rpt", synth_dir / "timing.rpt"
    if not util_rpt.exists() or not timing_rpt.exists():
        raise StageError("synth", f"reports not found in {synth_dir}; run synth/run_synth.sh first")
    wns = vivado.parse_wns(timing_rpt.read_text())
    values = vivado.parse_utilization(util_rpt.read_text())
    values |= {
        "tool": "vivado",
        "top": tc.fpga.synth_top.format(top=top),
        "part": tc.fpga.part,
        "clock_mhz": tc.fpga.clock_mhz,
        "wns_ns": wns,
        "fmax_mhz": vivado.fmax_mhz(tc.fpga.period_ns, wns),
    }
    report.replace(ws.out, "synth", values)


def cmd_synth(args: argparse.Namespace) -> int:
    _, ws = _workspace(args)
    script = ws.out / "synth" / _SYNTH_SCRIPT[_toolchain(args).fpga.synth_tool]
    if not script.exists():
        raise StageError("synth", f"{script} not found; run `oasis tb` (see `oasis tools`)")
    vivado.run_script(script, "synth")
    _collect_synth(ws, args)
    print(report.summary(report.load(ws.out)))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Re-read whatever sim/check/synth outputs exist; never runs a tool."""
    bench, ws = _workspace(args)
    if (ws.out / "sim" / "sim.log").exists():
        _check(bench, ws)
    if (ws.out / "synth" / _SYNTH_RESULT[_toolchain(args).fpga.synth_tool]).exists():
        _collect_synth(ws, args)
    print(report.summary(report.load(ws.out)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Shorthand for `oasis compile <benchmark> --sim --synth`."""
    args.dump_all, args.tb, args.sim, args.synth = True, True, True, True
    args.stop_after = None
    args.pipeline = None
    return cmd_compile(args)


def cmd_list(args: argparse.Namespace) -> int:
    """List benchmarks and their sizes from models/data.py."""
    for name, entry in sorted(registry().items()):
        sizes = ", ".join(f"{k}={v}" for k, v in entry["inputs"].items())
        status = "[wip]" if entry.get("status") == "wip" else ""
        print(f"{name:<18} {status:<5} {entry['dtype']:<4} {entry['model']:<40} {sizes}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    """Print how each configured tool and path resolves."""
    tc = _toolchain(args)
    print(f"frontend runner: {tc.frontend_runner}")
    print(f"simulator:       {tc.simulator} (cycle limit {tc.cycle_limit})")
    print(f"fpga:            {tc.fpga.part} @ {tc.fpga.clock_mhz:g} MHz, top {tc.fpga.synth_top}")
    print(f"synthesis:       {tc.fpga.synth_tool} (yosys family {tc.fpga.yosys_family})")
    for key in sorted(tc.tools):
        try:
            where = tc.resolve(key)
        except ToolNotFoundError:
            where = "NOT FOUND"
        print(f"  {key:<18} {where}")
    for key in sorted(tc.paths):
        try:
            where = str(tc.path(key))
        except ToolNotFoundError:
            where = "NOT FOUND"
        print(f"  {key:<18} {where}")
    return 0


def _synth_tool_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--synth-tool",
        choices=["yosys", "vivado"],
        help="synthesis backend (overrides environment/config; default: yosys)",
    )


def _simulator_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--simulator",
        choices=["xsim", "verilator"],
        help="simulator to run (default: [sim] simulator in config/toolchain.toml)",
    )


def _bench_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("benchmark", help="benchmark name from models/data.py (see `oasis list`)")
    p.add_argument(
        "--size", default="small", help="input size from models/data.py (default: small)"
    )
    p.add_argument("--out", help="output directory (default: out/<benchmark>_<size>)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oasis", description="PyTorch -> RTL compiler")
    parser.add_argument("--version", action="version", version=f"oasis {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compile", help="compile a benchmark through the pipeline")
    _bench_args(p)
    p.add_argument(
        "--dump-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write every stage's IR as NN_<stage>.* (default: on)",
    )
    p.add_argument(
        "--tb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate the testbench and simulation/synthesis scripts after compiling (default: on)",
    )
    p.add_argument(
        "--sim",
        action="store_true",
        help="then run Vivado xsim (sim/run_xsim.sh) and check against golden (slow)",
    )
    _simulator_arg(p)
    p.add_argument(
        "--synth",
        "--syn",
        dest="synth",
        action="store_true",
        help="then run synthesis ([fpga] synth_tool: vivado or yosys) and report",
    )
    p.add_argument("--pipeline", help="pipeline TOML (default: config/pipelines/v0.toml)")
    p.add_argument("--top", default="forward", help="top-level function name (default: forward)")
    p.add_argument("--stop-after", metavar="STAGE", help="stop after the named stage")
    _synth_tool_arg(p)
    p.set_defaults(func=cmd_compile)

    for name, func, help_text in (
        ("tb", cmd_tb, "generate testbench, memory files, simulation and synthesis scripts"),
        ("sim", cmd_sim, "run the simulation script (xsim or verilator), then check"),
        ("check", cmd_check, "compare the simulated output with golden.npy"),
        ("synth", cmd_synth, "run the synthesis script (vivado or yosys), then parse reports"),
        ("report", cmd_report, "re-read existing logs/reports and print the summary"),
    ):
        p = sub.add_parser(name, help=help_text)
        _bench_args(p)
        if name in ("tb", "synth", "report"):
            _synth_tool_arg(p)
        if name == "sim":
            _simulator_arg(p)
        p.set_defaults(func=func)

    p = sub.add_parser("run", help="same as `oasis compile <benchmark> --sim --synth`")
    _bench_args(p)
    _simulator_arg(p)
    p.add_argument("--top", default="forward", help="top-level function name (default: forward)")
    _synth_tool_arg(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("list", help="list benchmarks from models/data.py")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("tools", help="show how configured external tools resolve")
    _synth_tool_arg(p)
    p.set_defaults(func=cmd_tools)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (
        StageError,
        ToolNotFoundError,
        MemMapError,
        FileNotFoundError,
        AttributeError,
        KeyError,
        ValueError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
