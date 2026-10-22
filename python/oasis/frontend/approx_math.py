"""Frontend-only scalar f32 exponential approximation.

Evaluate the degree-10 Taylor polynomial at x/128 using Horner's rule, then
square seven times. Range reduction keeps the polynomial argument in
[-104/128, 89/128]. Values below -104 round to zero in f32; values above
89 overflow. NaNs propagate through arithmetic, and infinities are selected
explicitly. This is an approximation, not a correctly rounded libm exp.

Also fold constant f64-to-f32 truncation (e.g. attention's scale), including
inexact conversions that upstream canonicalization leaves in place.

Only arith operations are emitted: no LLVM ABI, integer/float conversions,
or backend linkage to torch-mlir. Unsupported types deliberately fail.
"""

from math import factorial

PASS_NAME = "oasis-approx-math"


def approximate_math(module) -> None:
    from torch_mlir import ir
    from torch_mlir.dialects import arith

    def visit(operation):
        for region in operation.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    visit(child.operation)
        if operation.name == "arith.truncf" and operation.results[0].type == ir.F32Type.get():
            source = operation.operands[0].owner
            if isinstance(source, ir.OpView):
                source = source.operation
            if isinstance(source, ir.Operation) and source.name == "arith.constant":
                attr = source.attributes["value"]
                if operation.operands[0].type == ir.F64Type.get():
                    with ir.InsertionPoint(operation), operation.location:
                        value = ir.FloatAttr(attr).value
                        replacement = arith.ConstantOp(
                            ir.F32Type.get(), ir.FloatAttr.get(ir.F32Type.get(), value)
                        ).result
                        operation.results[0].replace_all_uses_with(replacement)
                    operation.erase()
            return
        if operation.name != "math.exp":
            return
        if operation.results[0].type != ir.F32Type.get():
            raise ValueError("oasis-approx-math supports only scalar f32 math.exp")
        with ir.InsertionPoint(operation), operation.location:
            f32 = ir.F32Type.get()

            def constant(value):
                return arith.ConstantOp(f32, ir.FloatAttr.get(f32, value)).result

            x = operation.operands[0]
            low = arith.CmpFOp(arith.CmpFPredicate.OLT, x, constant(-104.0)).result
            high = arith.CmpFOp(arith.CmpFPredicate.OGT, x, constant(89.0)).result
            # Keep intermediate arithmetic finite for +/-infinity and huge inputs.
            bounded = arith.SelectOp(low, constant(-104.0), x).result
            bounded = arith.SelectOp(high, constant(89.0), bounded).result
            r = arith.MulFOp(bounded, constant(1.0 / 128)).result
            p = constant(1.0 / factorial(10))
            for degree in range(9, -1, -1):
                p = arith.AddFOp(
                    arith.MulFOp(p, r).result, constant(1.0 / factorial(degree))
                ).result
            for _ in range(7):
                p = arith.MulFOp(p, p).result
            p = arith.SelectOp(low, constant(0.0), p).result
            p = arith.SelectOp(high, constant(float("inf")), p).result
            operation.results[0].replace_all_uses_with(p)
        operation.erase()

    visit(module.operation)
    module.operation.verify()
