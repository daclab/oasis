"""Benchmark registry: which model class to build and the shapes of its inputs.

Model files (models/<suite>/<name>.py) contain only the torch.nn.Module.
Inputs are generated from these shapes by oasis.inputs (seeded), and outputs are
checked by oasis.verify. Each entry:
    "model":   "<path under models/>:<class name>"
    "dtype":   "f32" | "i32"
    "inputs":  {size: [shape of each activation input of forward(), in order]}
    "init":    {size: constructor keyword arguments}                       (optional)
    "weights": "args" = parameters/buffers become kernel arguments, i.e.
               external memories (default "inline")                      (optional)
"""

BENCHMARKS = {
    "gemm": {
        "model": "polybench/gemm.py:gemm",
        "dtype": "f32",
        # C = A @ B with A[NI][NK], B[NK][NJ]
        "inputs": {
            "small": [(32, 32), (32, 32)],
            "medium": [(200, 240), (240, 220)],  # PolyBench MEDIUM sizes
        },
    },
    # ---- Ported from From-Algorithm-to-RTL/examples (same shapes as the Allo versions) ----
    "relu": {
        "model": "nn/relu.py:relu",
        "dtype": "f32",
        "inputs": {"small": [(1, 3, 10, 10), (1, 3, 10, 10)]},
    },
    "ffnn": {
        "model": "nn/ffnn.py:ffnn",
        "dtype": "f32",
        "weights": "args",  # l1/l2 weights and biases are memories, as in the Allo version
        "inputs": {"small": [(48, 64)]},
    },
    "increment": {
        "model": "misc/increment.py:increment",
        "dtype": "i32",
        "inputs": {"small": [(1,)]},
    },
}
