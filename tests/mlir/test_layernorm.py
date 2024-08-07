# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

import numpy as np

def layer_norm(x):
    eps = 0.00001
    gamma = 0.00001
    beta = 0.00001
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta

def transformer_block(x):  # [n_seq, n_embd] -> [n_seq, n_embd]
    return layer_norm(x)

def gpt2(x, p_x):
    vocab_size = 50257
    n_positions = 1024
    n_embed = 768
    wte = np.random.randn(vocab_size, n_embed)
    wpe = np.random.randn(n_positions, n_embed)
    x = wte[x]
    x = x + wpe[p_x]
    x = transformer_block(x)

    return x


np.random.seed(42)
inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
pos_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = gpt2(inputs, pos_array)


# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %34 = "arith.constant" () {value = 42: i32} : () -> i32
# CHECK-NEXT:   %35 = "numpy.random.seed" (%34) : (i32) -> ()
# CHECK-NEXT:   %36 = "arith.constant" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %37 = "arith.constant" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %83 = "arith.constant" () {value = 50257: i32} : () -> i32
# CHECK-NEXT:   %84 = "arith.constant" () {value = 1024: i32} : () -> i32
# CHECK-NEXT:   %85 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %86 = "numpy.random.randn" (%83,%85) : (i32,i32) -> tensor<50257x768xf32>
# CHECK-NEXT:   %87 = "numpy.random.randn" (%84,%85) : (i32,i32) -> tensor<1024x768xf32>
# CHECK-NEXT:   %89 = "tosa.gather" (%86,%36) : (tensor<50257x768xf32>,tensor<16xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %91 = "tosa.gather" (%87,%37) : (tensor<1024x768xf32>,tensor<16xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %92 = "tosa.add" (%89,%91) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %95 = "arith.constant" () {value = 0.00001000: f32} : () -> f32
# CHECK-NEXT:   %96 = "arith.constant" () {value = 0.00001000: f32} : () -> f32
# CHECK-NEXT:   %97 = "arith.constant" () {value = 0.00001000: f32} : () -> f32
# CHECK-NEXT:   %98 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %99 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %117 = "tosa.reduce_sum" (%92) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %118 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %119 = "tosa.cast" (%118) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %120 = "tosa.reciprocal" (%119) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %121 = "tosa.mul" (%117,%120) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %122 = "tosa.reshape" (%121) {new_shape = [16, 1]} : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %101 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %102 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %123 = "tosa.reduce_sum" (%92) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %124 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %125 = "tosa.cast" (%124) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %126 = "tosa.reciprocal" (%125) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %127 = "tosa.mul" (%123,%126) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %128 = "tosa.reshape" (%127) {new_shape = [16, 1]} : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %129 = "tosa.cast" (%128) : (tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %130 = "tosa.sub" (%92,%129) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %131 = "tosa.mul" (%130,%130) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %132 = "tosa.reduce_sum" (%92) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %133 = "tosa.mul" (%132,%126) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %134 = "tosa.reshape" (%133) {new_shape = [16, 1]} : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %104 = "tosa.sub" (%92,%122) : (tensor<16x768xf32>,tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %105 = "tensor.splat" (%96) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %106 = "tosa.mul" (%105,%104) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %107 = "tensor.splat" (%95) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %108 = "tosa.add" (%134,%107) : (tensor<16x1xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %116 = "math.sqrt" (%108) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %110 = "tosa.reciprocal" (%116) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %111 = "tosa.mul" (%106,%110) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %112 = "tensor.splat" (%97) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %113 = "tosa.add" (%111,%112) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT: }) : () -> ()
