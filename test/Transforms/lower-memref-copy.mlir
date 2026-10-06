// RUN: oasis-opt %s --oasis-lower-memref-copy | FileCheck %s
// RUN: oasis-opt %s --oasis-lower-memref-copy --lower-affine --fold-memref-alias-ops --lower-affine | FileCheck %s --check-prefix=FOLDED

// Zero padding as produced by bufferizing tensor.pad: copy into a strided subview.
// CHECK-LABEL: func.func @pad_copy
// CHECK-NOT:     memref.copy
// CHECK:         affine.for %[[I:.*]] = 0 to 2 {
// CHECK-NEXT:      affine.for %[[J:.*]] = 0 to 3 {
// CHECK-NEXT:        %[[V:.*]] = affine.load %{{.*}}[%[[I]], %[[J]]] : memref<2x3xf32>
// CHECK-NEXT:        affine.store %[[V]], %{{.*}}[%[[I]], %[[J]]] : memref<2x3xf32, strided<[5, 1], offset: 6>>

// FOLDED-LABEL: func.func @pad_copy
// FOLDED-NOT:     memref.subview
// FOLDED-NOT:     memref.copy
// FOLDED-NOT:     affine.
// FOLDED:         memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x5xf32>
func.func @pad_copy(%src: memref<2x3xf32>, %dst: memref<4x5xf32>) {
  %sv = memref.subview %dst[1, 1] [2, 3] [1, 1] : memref<4x5xf32> to memref<2x3xf32, strided<[5, 1], offset: 6>>
  memref.copy %src, %sv : memref<2x3xf32> to memref<2x3xf32, strided<[5, 1], offset: 6>>
  return
}

// CHECK-LABEL: func.func @plain_copy
// CHECK-NOT:     memref.copy
// CHECK-COUNT-3: affine.for
func.func @plain_copy(%src: memref<2x3x4xi32>, %dst: memref<2x3x4xi32>) {
  memref.copy %src, %dst : memref<2x3x4xi32> to memref<2x3x4xi32>
  return
}
