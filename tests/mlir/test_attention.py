# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v, mask):
    n_embed = 768
    n_head = 12
    n_embed_per_head = n_embed / n_head
    kt = np.transpose(k)
    qkt = q @ kt
    scaled_qkt = softmax(qkt / np.sqrt(n_embed_per_head) + mask)
    return scaled_qkt @ v

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   func.func @softmax(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %0 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %1 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %2 = "numpy.max" (%arg0,%0,%1) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %3 = "tosa.sub" (%arg0,%2) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %4 = "numpy.exp" (%3) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %5 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %6 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %7 = "numpy.sum" (%4,%5,%6) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %8 = "tosa.reciprocal" (%7) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %9 = "tosa.mul" (%4,%8) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %9: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @attention(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %11 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %12 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:     %13 = "arith.divsi" (%11,%12) : (i32,i32) -> i32
# CHECK-NEXT:     %14 = "numpy.transpose" (%arg1) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %15 = "tosa.matmul" (%arg0,%14) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %16 = "numpy.sqrt" (%13) : (i32) -> i32
# CHECK-NEXT:     %17 = "tensor.splat" (%16) : (i32) -> tensor<*xf32>
# CHECK-NEXT:     %18 = "tosa.reciprocal" (%17) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %19 = "tosa.mul" (%15,%18) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %20 = "tosa.add" (%19,%arg3) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %21 = func.call @softmax(%20) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %22 = "tosa.matmul" (%21,%arg2) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %22: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }) : () -> ()
