//===- Passes.h - OASIS transformation passes -------------------*- C++ -*-===//

#ifndef OASIS_TRANSFORMS_PASSES_H
#define OASIS_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace oasis {

#define GEN_PASS_DECL
#include "oasis/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "oasis/Transforms/Passes.h.inc"

} // namespace oasis

#endif // OASIS_TRANSFORMS_PASSES_H
