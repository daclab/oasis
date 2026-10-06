//===- LowerMemrefCopy.cpp - memref.copy -> affine loop nests -------------===//
//
// Ported from flux-opt's lower-memref-copy-to-affine (From-Algorithm-to-RTL).
//
//===----------------------------------------------------------------------===//

#include "oasis/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace oasis {
#define GEN_PASS_DEF_LOWERMEMREFCOPY
#include "oasis/Transforms/Passes.h.inc"
} // namespace oasis

using namespace mlir;

namespace {

/// Build `rank` nested affine.for loops; the innermost body copies one element.
void buildCopyNest(OpBuilder &builder, Location loc, Value src, Value dst,
                   ArrayRef<int64_t> shape, SmallVectorImpl<Value> &ivs) {
  if (ivs.size() == shape.size()) {
    Value v = affine::AffineLoadOp::create(builder, loc, src, ivs);
    affine::AffineStoreOp::create(builder, loc, v, dst, ivs);
    return;
  }
  auto forOp = affine::AffineForOp::create(builder, loc, 0, shape[ivs.size()]);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(forOp.getBody());
  ivs.push_back(forOp.getInductionVar());
  buildCopyNest(builder, loc, src, dst, shape, ivs);
  ivs.pop_back();
}

LogicalResult lowerCopy(memref::CopyOp copy) {
  auto srcType = dyn_cast<MemRefType>(copy.getSource().getType());
  auto dstType = dyn_cast<MemRefType>(copy.getTarget().getType());
  if (!srcType || !dstType)
    return copy.emitError("expected ranked memref operands");
  if (srcType.getElementType() != dstType.getElementType())
    return copy.emitError("source and target element types differ");
  if (srcType.getShape() != dstType.getShape())
    return copy.emitError("source and target shapes differ");
  if (!srcType.hasStaticShape())
    return copy.emitError("dynamic shapes are not supported");

  OpBuilder builder(copy);
  SmallVector<Value> ivs;
  buildCopyNest(builder, copy.getLoc(), copy.getSource(), copy.getTarget(),
                srcType.getShape(), ivs);
  copy.erase();
  return success();
}

struct LowerMemrefCopyPass
    : public oasis::impl::LowerMemrefCopyBase<LowerMemrefCopyPass> {
  void runOnOperation() override {
    SmallVector<memref::CopyOp> copies;
    getOperation().walk([&](memref::CopyOp op) { copies.push_back(op); });
    for (memref::CopyOp copy : copies)
      if (failed(lowerCopy(copy)))
        return signalPassFailure();
  }
};

} // namespace
