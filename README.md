# OASIS

**Open Infrastructure for Automated Synthesis of AI Models to Hardware**

OASIS is an open-source compiler that lowers PyTorch models to synthesizable RTL for FPGAs and ASICs. It is built on [MLIR](https://mlir.llvm.org/) and [CIRCT](https://circt.llvm.org/), and it keeps the whole flow inside a multi-level IR stack instead of handing generated C/C++ to a high-level synthesis (HLS) tool. Optimization intent such as tiling, memory layout, parallelism and precision is carried as explicit IR structure and attributes, so later stages don't have to re-infer it from general-purpose code.

> **Status: early development (v0.1).** One complete, tested path works: a small PyTorch model (`gemm`, f32) to SystemVerilog, simulated in Vivado and checked against PyTorch. Interfaces and pass pipelines will change.

---

## Pipeline

```
 PyTorch model
      │  torch.export + torch-mlir FX importer
      ▼
 torch dialect ──▶ linalg-on-tensors
      │  one-shot bufferization, results → out-params
      ▼
 linalg on memrefs ──▶ affine loops ──▶ scf + memref + arith
      │
 ═════╪═════  handoff: textual .mlir on disk  ═════════════
      │
      ▼  oasis-opt: flatten memories to 1-D (OASIS pass)
 scf + memref + arith, 1-D memories
      │
      ▼  CIRCT
 scf ──▶ Calyx ──▶ native Calyx (.futil)
      │  Calyx compiler
      ▼
 SystemVerilog ──▶ Vivado xsim simulation ──▶ check vs. PyTorch (bit-exact for integers)
              └──▶ Vivado synthesis       ──▶ area, timing
```

Every stage writes its output to disk as a numbered file (`00_linalg.mlir`, `01_bufferized.mlir`, … `07_design.sv`), so you can inspect the design at any level of abstraction.

## Requirements

| Dependency | Version tested | Used for |
|---|---|---|
| Linux, Python 3.11 (conda) | Ubuntu 22.04 | everything |
| torch-mlir (nightly wheel) + PyTorch nightly | torch-mlir 20261001, torch 2.15.0.dev20261002 | frontend |
| CIRCT, with its bundled LLVM/MLIR | commit `a8cf045b3` | `hlstool`, `circt-translate`; LLVM/MLIR for building `oasis-opt` |
| Calyx compiler (Rust) | built from source | Calyx to SystemVerilog |
| CMake ≥ 3.20, Ninja, a C++17 compiler | CMake 3.22 | building CIRCT and `oasis-opt` |
| AMD Vivado (optional) | 2023.2 | xsim simulation and synthesis |
| Verilator ≥ 5 (optional) | — | faster simulation of large designs (`conda install -n oasis -c conda-forge verilator`) |

## Installation

### 1. Python environment and torch-mlir

```bash
conda create -n oasis python=3.11
conda activate oasis
pip install --pre torch-mlir torchvision \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

### 2. CIRCT (with LLVM/MLIR)

CIRCT ships LLVM as a submodule. Build LLVM/MLIR first, then CIRCT (this takes a while):

```bash
git clone https://github.com/llvm/circt.git ~/circt
cd ~/circt
git checkout a8cf045b3          # the commit OASIS v0.1 was tested with
git submodule update --init

# LLVM + MLIR
mkdir -p llvm/build && cd llvm/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja

# CIRCT
cd ~/circt && mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$HOME/circt/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/circt/llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja
```

This gives `~/circt/build/bin/{circt-opt,hlstool,circt-translate}` and the LLVM/MLIR build in `~/circt/llvm/build`.

### 3. Calyx

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # if Rust is not installed
git clone https://github.com/calyxir/calyx.git ~/calyx
cd ~/calyx
cargo build --release            # binary: ~/calyx/target/release/calyx
```

OASIS also needs the repository path itself, because the Calyx primitives live there (`calyx -l ~/calyx`).

### 4. OASIS

```bash
git clone <this repository> ~/OASIS
cd ~/OASIS
pip install -e ".[dev]"          # installs the `oasis` command into the conda env

# oasis-opt (C++), built against CIRCT's LLVM/MLIR
cmake -G Ninja -S . -B build/cmake \
  -DMLIR_DIR=$HOME/circt/llvm/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=$HOME/circt/llvm/build/bin/llvm-lit
ninja -C build/cmake                  # → build/cmake/bin/oasis-opt
ninja -C build/cmake check-oasis      # C++ pass tests
```

### 5. Tell OASIS where the tools are

```bash
cp config/toolchain.local.toml.example config/toolchain.local.toml
# edit the paths: circt-opt, hlstool, circt-translate, calyx, calyx_lib, and Vivado if you have it
oasis tools                      # shows how every tool resolves; anything "NOT FOUND" needs a path
```

`config/toolchain.local.toml` is machine-specific and not committed. Any entry can also be set with an environment variable, e.g. `OASIS_CIRCT_OPT=/path/to/circt-opt`.

## Usage

Activate the environment first (`conda activate oasis`) and run from the repository root.

| Command | What it does |
|---|---|
| `oasis list` | List the benchmarks and their input sizes |
| `oasis compile gemm` | PyTorch → RTL (`07_design.sv`), every stage's IR, testbench and Vivado scripts |
| `oasis compile gemm --size medium` | Same, with another input size from `models/data.py` |
| `oasis compile gemm --sim` | ...then simulate in Vivado xsim and compare with PyTorch |
| `oasis compile gemm --sim --simulator verilator` | Same, simulating with Verilator (much faster on large designs) |
| `oasis compile gemm --synth` | ...then run Vivado synthesis and report LUT/FF/DSP/BRAM, WNS, Fmax (`--syn` also works) |
| `oasis compile gemm --sim --synth` | Both (same as `oasis run gemm`) |
| `oasis compile gemm --no-tb` | Compile only, no testbench |
| `oasis compile gemm --no-dump-all` | Keep only the files later stages need |
| `oasis compile gemm --stop-after scf` | Stop after a named stage (e.g. only the frontend) |
| `oasis tb gemm` | Regenerate the testbench and Vivado scripts only |
| `oasis check gemm` | Compare an existing xsim run's output with PyTorch |
| `oasis report gemm` | Re-read existing simulation/synthesis results and print the summary |
| `oasis tools` | Show where every external tool resolves |

The Vivado runs can also be started by hand. `oasis compile` prints the exact commands:

```bash
bash out/gemm_small/sim/run_xsim.sh      && oasis check gemm     # simulation + check
bash out/gemm_small/synth/run_synth.sh   && oasis report gemm    # synthesis + report
```

### Results

All outputs of a run go to `out/<benchmark>_<size>/`:

| Path | Contents |
|---|---|
| `NN_<stage>.*` | IR of every stage, ending with `07_design.sv` (the RTL) |
| `inputs.npz`, `golden.npy` | Inputs and the expected output from PyTorch |
| `sim/` | Testbench, memory files, xsim logs (`sim.log`) and simulated output |
| `synth/` | Synthesis RTL and scripts, `utilization.rpt`, `timing.rpt` |
| `report.json` | Summary: cycles, check result, resources, timing |
| `logs/` | The command and output of each stage |

## Benchmarks

| Benchmark | Data type | Description | v0.1 status |
|---|---|---|---|
| `gemm` | f32 | `C = A @ B`; 32×32 (`small`), PolyBench MEDIUM (`medium`) | RTL, simulated (PASS), synthesized |
| `relu` | f32 | ReLU(x + y), 1×3×10×10 | RTL and testbench |
| `ffnn` | f32 | Linear 64→48, ReLU, Linear 48→4; weights as memories | RTL and testbench |
| `increment` | i32 | a + 1 | RTL and testbench |

### Adding a benchmark

1. Put the model in `models/<suite>/<name>.py`. The file contains only the `torch.nn.Module` class.
2. Register it in `models/data.py`: the class, its data type, and the input shapes per size. Optionally add constructor arguments per size (`init`) and weights as memories (`weights: "args"`).
3. Run `oasis compile <name>`.

OASIS generates the inputs from the registered shapes and computes the golden output with PyTorch; the model file never contains inputs or checks.

## Configuration

| File | Contents |
|---|---|
| `config/toolchain.toml` | Default tool names, simulation cycle limit, FPGA target (`[fpga] part`, `clock_mhz`, `synth_top`) |
| `config/toolchain.local.toml` | Your machine's tool paths; overrides the defaults (not committed) |
| `config/pipelines/v0.toml` | The ordered stages and the exact passes/flags of each tool |
| `models/data.py` | Benchmark registry |

To target another FPGA for one run: `OASIS_FPGA_PART=xc7z020clg484-1 OASIS_CLOCK_MHZ=100 oasis compile gemm --synth`.

## Repository layout

```
OASIS/
├── python/oasis/                 the `oasis` command
│   ├── cli.py                    commands: compile, tb, check, report, list, tools
│   ├── config.py                 toolchain + pipeline config loading
│   ├── stages.py                 stage runner, numbered dumps, hand-off check
│   ├── models.py                 benchmark registry loading
│   ├── inputs.py                 seeded input generation
│   ├── verify.py                 golden output (PyTorch) and comparison
│   ├── data.py                   numpy <-> hex memory files
│   ├── report.py                 report.json summary
│   ├── frontend/
│   │   ├── export.py             PyTorch -> linalg (torch-mlir FX importer)
│   │   ├── lower.py              linalg -> scf/memref/arith
│   │   └── prepare.py            weights as kernel arguments
│   └── backend/
│       ├── memmap.py             which Calyx memory holds which argument
│       ├── testbench.py          sim/: tb.sv + memory init files
│       ├── vivado.py             xsim/synthesis scripts, report parsing
│       └── templates/            tb.sv, run_xsim.sh, synth.tcl, clocks.xdc, run_synth.sh
├── include/oasis/Transforms/     oasis-opt pass definitions (Passes.td, Passes.h)
├── lib/Transforms/               oasis-opt passes (C++)
├── tools/oasis-opt/              oasis-opt driver
├── models/
│   ├── data.py                   benchmark registry: model, dtype, input shapes
│   ├── polybench/gemm.py
│   ├── nn/                       relu.py, ffnn.py
│   └── misc/increment.py
├── config/
│   ├── toolchain.toml            default tools, FPGA target, simulation settings
│   ├── toolchain.local.toml.example
│   └── pipelines/v0.toml         stages and the passes/flags of each tool
├── tests/                        Python tests (python -m pytest -q)
├── test/                         oasis-opt pass tests (ninja -C build/cmake check-oasis)
├── CMakeLists.txt                oasis-opt build
└── pyproject.toml                Python package and the `oasis` command
```

Generated, not committed: `out/<benchmark>_<size>/` (compile, simulation and synthesis outputs) and `build/` (the `oasis-opt` build).

## Acknowledgments

OASIS builds on LLVM/MLIR, CIRCT, torch-mlir, Calyx, Verilator and Yosys.

NSF-Funded Research Project | NSF Award #2608702

## License

OASIS is released under the MIT License.
