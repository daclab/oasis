// RUN: oasis-opt %s --oasis-flatten-memref --split-input-file | FileCheck %s
// RUN: oasis-opt %s --oasis-flatten-memref --canonicalize --split-input-file | FileCheck %s --check-prefix=CANON

// 4-D, as in conv padding: element [0, c, h, w] of 1x3x22x22 is c*484 + h*22 + w.
// Horner form: ((i0*3 + i1)*22 + i2)*22 + i3. (CIRCT's flatten-memref multiplies by
// 1452, 484, 22 here, which is wrong.)
// CHECK-LABEL: func.func @pad_store(
// CHECK-SAME:    %[[M:.*]]: memref<1452xf32>, %[[C:.*]]: index, %[[H:.*]]: index, %[[W:.*]]: index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[A0:.*]] = arith.muli %[[C0]], %[[C3]] : index
// CHECK:         %[[A1:.*]] = arith.addi %[[A0]], %[[C]] : index
// CHECK:         %[[C22:.*]] = arith.constant 22 : index
// CHECK:         %[[A2:.*]] = arith.muli %[[A1]], %[[C22]] : index
// CHECK:         %[[A3:.*]] = arith.addi %[[A2]], %[[H]] : index
// CHECK:         %[[C22B:.*]] = arith.constant 22 : index
// CHECK:         %[[A4:.*]] = arith.muli %[[A3]], %[[C22B]] : index
// CHECK:         %[[A5:.*]] = arith.addi %[[A4]], %[[W]] : index
// CHECK:         memref.store %{{.*}}, %[[M]][%[[A5]]] : memref<1452xf32>
func.func @pad_store(%m: memref<1x3x22x22xf32>, %c: index, %h: index, %w: index) {
  %c0 = arith.constant 0 : index
  %v = arith.constant 1.0 : f32
  memref.store %v, %m[%c0, %c, %h, %w] : memref<1x3x22x22xf32>
  return
}

// -----

// Power-of-two sizes use shifts: [i, j] of 4x8 is (i << 3) + j.
// CHECK-LABEL: func.func @pow2_load(
// CHECK-SAME:    %[[M:.*]]: memref<32xi32>, %[[I:.*]]: index, %[[J:.*]]: index
// CHECK:         %[[S:.*]] = arith.constant 3 : index
// CHECK:         %[[SH:.*]] = arith.shli %[[I]], %[[S]] : index
// CHECK:         %[[A:.*]] = arith.addi %[[SH]], %[[J]] : index
// CHECK:         memref.load %[[M]][%[[A]]] : memref<32xi32>
func.func @pow2_load(%m: memref<4x8xi32>, %i: index, %j: index) -> i32 {
  %v = memref.load %m[%i, %j] : memref<4x8xi32>
  return %v : i32
}

// -----

// Buffers, rank 0, calls and returns are flattened too; 1-D memrefs are untouched.
// CHECK-LABEL: func.func @callee(%{{.*}}: memref<6xf32>, %{{.*}}: memref<1xf32>) -> memref<4xf32>
// CHECK-LABEL: func.func @caller(
// CHECK:         %[[B:.*]] = memref.alloc() : memref<6xf32>
// CHECK:         %[[S:.*]] = memref.alloca() : memref<1xf32>
// CHECK:         memref.load %[[S]][%{{.*}}] : memref<1xf32>
// CHECK:         call @callee(%[[B]], %[[S]]) : (memref<6xf32>, memref<1xf32>) -> memref<4xf32>
// CHECK:         memref.dealloc %[[B]] : memref<6xf32>
func.func @callee(%a: memref<2x3xf32>, %s: memref<f32>) -> memref<4xf32> {
  %r = memref.alloc() : memref<4xf32>
  return %r : memref<4xf32>
}
func.func @caller() -> f32 {
  %b = memref.alloc() {alignment = 64 : i64} : memref<2x3xf32>
  %s = memref.alloca() : memref<f32>
  %v = memref.load %s[] : memref<f32>
  %r = call @callee(%b, %s) : (memref<2x3xf32>, memref<f32>) -> memref<4xf32>
  memref.dealloc %b : memref<2x3xf32>
  return %v : f32
}

// CANON-LABEL: func.func @pad_store(
// CANON:         %[[C484:.*]] = arith.constant 22 : index
// CANON-NOT:     arith.constant 1452
// CANON-NOT:     arith.constant 484
