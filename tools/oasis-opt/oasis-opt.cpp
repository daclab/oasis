//===- oasis-opt.cpp - OASIS optimizer driver -----------------------------===//
//
// The OASIS passes plus the upstream MLIR passes the backend pipeline uses.
// Backend side: links CIRCT's LLVM/MLIR only. Kept deliberately small (no
// registerAllDialects/registerAllPasses) so the binary links quickly.
//
//===----------------------------------------------------------------------===//

#include "oasis/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();

  mlir::registerTransformsPasses();      // canonicalize, cse, ...
  mlir::registerLowerAffinePass();       // lower-affine
  mlir::memref::registerMemRefPasses();  // fold-memref-alias-ops, ...
  mlir::arith::registerArithPasses();    // arith-expand, ...
  oasis::registerOASISTransformsPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "OASIS optimizer driver\n", registry));
}
