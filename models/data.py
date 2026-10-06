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
    "prepare": steps from oasis.frontend.prepare, in order               (optional)
"""

# Conv nets: random BN statistics (so folding is really exercised), then fold BN into
# the convs, and pass all weights in as memories.
_CNN = {"dtype": "f32", "weights": "args", "prepare": ["randomize_bn", "fold_bn"]}

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
    # GELU lowers to math.erf, which has no hardware form yet: the hand-off check rejects it
    # until a math-approximation pass exists.
    "gelu": {
        "model": "nn/gelu.py:gelu",
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
    # Full ResNet-18: compile + synthesis. Simulation is very long (~0.8 G cycles at 32x32).
    "resnet18": {
        **_CNN,
        "model": "cnn/resnet18.py:resnet18",
        "inputs": {"small": [(1, 3, 32, 32)], "large": [(1, 3, 224, 224)]},
    },
    # ResNet-18 pieces. "small": reduced channels so xsim finishes in minutes;
    # "medium": the real ResNet-18 sizes for a 32x32 image (long simulation).
    "resnet_stem": {
        **_CNN,
        "model": "cnn/resnet_blocks.py:stem",
        "inputs": {"small": [(1, 3, 16, 16)], "medium": [(1, 3, 32, 32)]},
        "init": {"small": {"channels": 8}, "medium": {"channels": 64}},
    },
    "resnet_block": {
        **_CNN,
        "model": "cnn/resnet_blocks.py:basic_block",
        "inputs": {"small": [(1, 8, 8, 8)], "medium": [(1, 64, 8, 8)]},
        "init": {"small": {"channels": 8}, "medium": {"channels": 64}},
    },
    "resnet_down": {
        **_CNN,
        "model": "cnn/resnet_blocks.py:down_block",
        "inputs": {"small": [(1, 8, 8, 8)], "medium": [(1, 64, 8, 8)]},
        "init": {
            "small": {"in_channels": 8, "out_channels": 16},
            "medium": {"in_channels": 64, "out_channels": 128},
        },
    },
    "resnet_head": {
        **_CNN,
        "model": "cnn/resnet_blocks.py:head",
        "inputs": {"small": [(1, 16, 2, 2)], "medium": [(1, 512, 1, 1)]},
        "init": {
            "small": {"channels": 16, "classes": 10},
            "medium": {"channels": 512, "classes": 1000},
        },
    },
}
