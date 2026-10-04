//===- FlattenMemref.cpp - memref<AxBxC...> -> memref<N> -------------------===//
//
// Correct replacement for CIRCT's flatten-memref (wrong addresses for rank >= 3).
//
//===----------------------------------------------------------------------===//

#include "oasis/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

namespace oasis {
#define GEN_PASS_DEF_FLATTENMEMREF
#include "oasis/Transforms/Passes.h.inc"
} // namespace oasis

using namespace mlir;

namespace {

/// Already flat: rank 1 with the identity layout.
bool isFlat(MemRefType type) {
  return type.getRank() == 1 && type.getLayout().isIdentity();
}

/// memref<AxBxC x T> -> memref<(A*B*C) x T>; rank 0 -> memref<1 x T>.
/// Returns null for shapes this pass can't flatten (dynamic or strided).
MemRefType flatType(MemRefType type) {
  if (!type.hasStaticShape() || !type.getLayout().isIdentity())
    return nullptr;
  return MemRefType::get({type.getNumElements()}, type.getElementType(),
                         MemRefLayoutAttrInterface(), type.getMemorySpace());
}

Value indexConstant(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIndexOp::create(b, loc, v);
}

/// Row-major address in Horner form: ((i0 * d1 + i1) * d2 + i2) ...
/// Each step multiplies by the *next dimension's size* (shift when it is a
/// power of two).
Value linearize(OpBuilder &b, Location loc, ValueRange indices,
                ArrayRef<int64_t> shape) {
  if (indices.empty())
    return indexConstant(b, loc, 0);
  Value addr = indices.front();
  for (size_t k = 1; k < indices.size(); ++k) {
    int64_t dim = shape[k];
    if (llvm::isPowerOf2_64(dim))
      addr = arith::ShLIOp::create(b, loc, addr,
                                   indexConstant(b, loc, llvm::Log2_64(dim)));
    else
      addr = arith::MulIOp::create(b, loc, addr, indexConstant(b, loc, dim));
    addr = arith::AddIOp::create(b, loc, addr, indices[k]);
  }
  return addr;
}

struct LoadOpFlatten : OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (!flatType(type))
      return rewriter.notifyMatchFailure(op, "dynamic or strided memref");
    Value addr = linearize(rewriter, op.getLoc(), adaptor.getIndices(),
                           type.getShape());
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getMemref(),
                                                ValueRange{addr});
    return success();
  }
};

struct StoreOpFlatten : OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = op.getMemRefType();
    if (!flatType(type))
      return rewriter.notifyMatchFailure(op, "dynamic or strided memref");
    Value addr = linearize(rewriter, op.getLoc(), adaptor.getIndices(),
                           type.getShape());
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getMemref(), ValueRange{addr});
    return success();
  }
};

template <typename AllocLikeOp>
struct AllocFlatten : OpConversionPattern<AllocLikeOp> {
  using OpConversionPattern<AllocLikeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AllocLikeOp op, typename AllocLikeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType flat = flatType(op.getType());
    if (!flat)
      return rewriter.notifyMatchFailure(op, "dynamic or strided memref");
    rewriter.replaceOpWithNewOp<AllocLikeOp>(op, flat);
    return success();
  }
};

struct DeallocFlatten : OpConversionPattern<memref::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, adaptor.getMemref());
    return success();
  }
};

struct FlattenMemrefPass
    : public oasis::impl::FlattenMemrefBase<FlattenMemrefPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    TypeConverter converter;
    converter.addConversion([](Type t) { return t; });
    converter.addConversion([](MemRefType t) -> std::optional<Type> {
      if (isFlat(t))
        return t;
      if (MemRefType flat = flatType(t))
        return flat;
      return std::nullopt; // not convertible: reported as a legalization error
    });

    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        return converter.isSignatureLegal(func.getFunctionType()) &&
               converter.isLegal(&func.getBody());
      return converter.isLegal(op);
    });

    RewritePatternSet patterns(ctx);
    patterns.add<LoadOpFlatten, StoreOpFlatten,
                 AllocFlatten<memref::AllocOp>, AllocFlatten<memref::AllocaOp>,
                 DeallocFlatten>(converter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                  converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
