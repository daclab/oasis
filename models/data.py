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
    "token_inputs": indices of forward() arguments that are token ids (int64 in
               [0, init["vocab_size"]) instead of random data)               (optional)
    "status":  "wip" = work in progress: not expected to compile end to end yet;
               "note" says why. `oasis list` shows it, `oasis compile` warns.  (optional)
"""

# Conv nets: random BN statistics (so folding is really exercised), then fold BN into
# the convs, and pass all weights in as memories.
_CNN = {"dtype": "f32", "weights": "args", "prepare": ["randomize_bn", "fold_bn"]}

# Transformers (WIP): weights as memories, like ffnn.
_TRANSFORMER = {"dtype": "f32", "weights": "args", "status": "wip"}
_SOFTMAX = "softmax lowers to math.exp (no hardware form yet; needs oasis-approx-math)"
_LAYERNORM = "LayerNorm needs a runtime 1/sqrt (math.rsqrt) of the input's variance"

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
    # PolyBench 3mm: E = A[NI][NK] @ B[NK][NJ], F = C[NJ][NM] @ D[NM][NL], G = E @ F.
    "k3mm": {
        "model": "polybench/k3mm.py:k3mm",
        "dtype": "f32",
        "status": "wip",
        "note": "small: Verilator check PASS 2026-10-09; medium and Vivado not run yet",
        "inputs": {
            # NI, NJ, NK, NL, NM = 16, 18, 20, 22, 24 (all different, to catch index mix-ups)
            "small": [(16, 20), (20, 18), (18, 24), (24, 22)],
            # PolyBench MEDIUM: NI, NJ, NK, NL, NM = 180, 190, 200, 210, 220
            "medium": [(180, 200), (200, 190), (190, 220), (220, 210)],
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
    # ---- Transformers (WIP) ----
    # Multi-head self-attention (Stream-HLS MultiHeadSelfAttention). medium = Stream-HLS's size.
    "attention": {
        **_TRANSFORMER,
        "model": "transformers/transformer.py:attention",
        "note": _SOFTMAX,
        "inputs": {"small": [(1, 16, 32)], "medium": [(1, 64, 128)]},
        "init": {
            "small": {"embed_dim": 32, "num_heads": 4},
            "medium": {"embed_dim": 128, "num_heads": 8},
        },
    },
    # Transformer block: LayerNorm + attention + LayerNorm + feed-forward (ReLU), residuals.
    "transformer_block": {
        **_TRANSFORMER,
        "model": "transformers/transformer.py:transformer_block",
        "note": f"{_SOFTMAX}; {_LAYERNORM}",
        "inputs": {"small": [(1, 16, 32)], "medium": [(1, 64, 128)]},
        "init": {
            "small": {"embed_dim": 32, "num_heads": 4, "ff_dim": 64},
            "medium": {"embed_dim": 128, "num_heads": 8, "ff_dim": 256},
        },
    },
    # Tiny decoder-only LLM: embeddings, causal blocks, LM head. Input: token ids.
    "tiny_llm": {
        **_TRANSFORMER,
        "model": "transformers/transformer.py:tiny_llm",
        "note": f"{_SOFTMAX}; {_LAYERNORM}; embedding lookup uses token ids as memory addresses",
        "token_inputs": [0],
        "inputs": {"small": [(1, 16)], "medium": [(1, 64)]},
        "init": {
            "small": {
                "vocab_size": 64,
                "max_seq": 16,
                "embed_dim": 32,
                "num_heads": 4,
                "ff_dim": 64,
                "num_layers": 2,
            },
            "medium": {
                "vocab_size": 256,
                "max_seq": 64,
                "embed_dim": 128,
                "num_heads": 8,
                "ff_dim": 256,
                "num_layers": 4,
            },
        },
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
