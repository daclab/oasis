"""Export a PyTorch model to linalg-on-tensors MLIR text with torch-mlir's FX importer."""

from __future__ import annotations

from oasis.stages import StageError


def export_linalg(model, inputs, func_name: str = "forward") -> tuple[str, str]:
    """(linalg-on-tensors MLIR, FX graphs) for `model` traced with `inputs`.

    The FX graphs are the readable torch.export graph (ATen ops) and the same graph after
    torch-mlir's decompositions, which is what FxImporter turns into MLIR.
    """
    try:
        import torch
        from torch_mlir import fx
        from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    except ImportError as e:
        raise StageError(
            "linalg",
            f"cannot import torch / torch_mlir ({e}). Install the torch-mlir nightly wheel "
            "(see README: Setup).",
        ) from e

    with torch.no_grad():
        # The steps of fx.export_and_import(model, ...), split so the graphs can be logged.
        prog = torch.export.export(model, tuple(inputs), strict=False)
        decomposed = prog.run_decompositions(get_decomposition_table())
        module = fx.export_and_import(
            decomposed,
            output_type="linalg-on-tensors",
            func_name=func_name,
            decomposition_table={},  # already decomposed
        )
    graphs = (
        "# FX graph from torch.export (ATen ops)\n"
        + prog.graph_module.print_readable(print_output=False)
        + "\n# FX graph after torch-mlir's decompositions (imported by FxImporter)\n"
        + decomposed.graph_module.print_readable(print_output=False)
    )
    return str(module), graphs
