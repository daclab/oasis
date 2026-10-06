//===- Passes.td - OASIS transformation passes -------------*- tablegen -*-===//

#ifndef OASIS_TRANSFORMS_PASSES
#define OASIS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LowerMemrefCopy : Pass<"oasis-lower-memref-copy", "::mlir::func::FuncOp"> {
  let summary = "Lower memref.copy to affine.for loop nests of load/store";
  let description = [{
    hlstool (SCF -> Calyx) cannot lower `memref.copy`. Bufferizing `tensor.pad`
    (zero padding before a convolution) produces copies into strided
    `memref.subview`s, so each copy becomes a loop nest that loads from the
    source and stores to the target. Run `lower-affine`, `fold-memref-alias-ops`
    and `lower-affine` afterwards to fold the subviews into plain index
    arithmetic on the underlying buffer.

    Ported from flux-opt's `lower-memref-copy-to-affine`
    (From-Algorithm-to-RTL/my-pass). Only static, identically shaped
    source/target memrefs are supported.
  }];
  let dependentDialects = ["::mlir::affine::AffineDialect",
                           "::mlir::memref::MemRefDialect"];
}

def FlattenMemref : Pass<"oasis-flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten statically shaped memrefs to 1-D with row-major addressing";
  let description = [{
    hlstool needs 1-D memories. Rewrites function arguments/results, calls,
    memref.alloc/alloca/dealloc and every memref.load/store so that each memref
    becomes memref<N x T> (rank 0 becomes memref<1 x T>), and each access uses
    the row-major address ((i0 * d1 + i1) * d2 + i2) ... with shifts for
    power-of-two sizes.

    Replaces CIRCT's `flatten-memref`, which computes wrong addresses for
    memrefs of rank 3 and higher (it multiplies the running Horner sum by the
    product of all remaining dimensions instead of the next dimension's size).
    Only identity-layout, statically shaped memrefs are supported; anything
    else is reported as an error.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::memref::MemRefDialect"];
}

#endif // OASIS_TRANSFORMS_PASSES
