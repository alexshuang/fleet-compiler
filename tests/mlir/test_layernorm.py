# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

import numpy as np


def layer_norm(x):
    eps = 0.00001
    gamma = 0.00001
    beta = 0.00001
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta


inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
layer_norm(inputs)


# CHECK: "builtin.module" () ({
# CHECK-NEXT:   func.func @layer_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %0 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %1 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %2 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %3 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %4 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %5 = "numpy.mean" (%arg0,%3,%4) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %6 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %7 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %8 = "numpy.var" (%arg0,%6,%7) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %9 = "tosa.sub" (%arg0,%5) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %10 = "tensor.splat" (%1) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %11 = "tosa.mul" (%10,%9) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %12 = "tensor.splat" (%0) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %13 = "tosa.add" (%8,%12) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %14 = "numpy.sqrt" (%13) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %15 = "tosa.reciprocal" (%14) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %16 = "tosa.mul" (%11,%15) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %17 = "tensor.splat" (%2) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %18 = "tosa.add" (%16,%17) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %18: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   %20 = "arith.constant" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %21 = func.call @layer_norm(%20) : (tensor<16xi32>) -> (tensor<*xf32>)
