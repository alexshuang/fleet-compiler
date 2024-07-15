import numpy as np

def layer_norm(x):
    eps = 0.00001
    gamma = 0.00001
    beta = 0.00001
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta

def gpt2(x, p_x):
    vocab_size = 50257
    n_positions = 1024
    n_embed = 768
    wte = np.random.randn(vocab_size, n_embed)
    wpe = np.random.randn(n_positions, n_embed)
    x = wte[x]
    x = x + wpe[p_x]
    # Not support for-loop, unroll it
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)
    return layer_norm(x) @ np.transpose(wte)

# gpt2-small config
# vocab_size = 50257
# n_positions = 1024
# n_embed = 768
# n_head = 12
# n_layers = 12
# bs = ctx_len = 16

inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
pos_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

np.random.seed(42)

# layer 0
# c_fc_w = np.random.randn(n_embed, 4 * n_embed)
# c_fc_b = np.random.randn(4 * n_embed)
# c_proj_w = np.random.randn(4 * n_embed, n_embed)
# c_proj_b = np.random.randn(n_embed)
# c_attn_w = np.random.randn(n_embed, 3 * n_embed)
# c_attn_b = np.random.randn(3 * n_embed)
# c_attn_proj_w = np.random.randn(n_embed, n_embed)
# c_attn_proj_b = np.random.randn(n_embed)

y = gpt2(inputs, pos_array)
# print(y)

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
# CHECK-NEXT:   func.func @gpt2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %20 = "arith.constant" () {value = 50257: i32} : () -> i32
# CHECK-NEXT:     %21 = "arith.constant" () {value = 1024: i32} : () -> i32
# CHECK-NEXT:     %22 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %23 = "numpy.random.randn" (%20,%22) : (i32,i32) -> tensor<50257x768xf32>
# CHECK-NEXT:     %24 = "numpy.random.randn" (%21,%22) : (i32,i32) -> tensor<1024x768xf32>
# CHECK-NEXT:     %25 = "tosa.gather" (%23,%arg0) : (tensor<50257x768xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %26 = "tosa.gather" (%24,%arg1) : (tensor<1024x768xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %27 = "tosa.add" (%25,%26) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %28 = func.call @layer_norm(%27) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %29 = "numpy.transpose" (%23) : (tensor<50257x768xf32>) -> tensor<768x50257xf32>
# CHECK-NEXT:     %30 = "tosa.matmul" (%28,%29) : (tensor<*xf32>,tensor<768x50257xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %30: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   %32 = "arith.constant" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %33 = "arith.constant" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %34 = "arith.constant" () {value = 42: i32} : () -> i32
# CHECK-NEXT:   %35 = "numpy.random.seed" (%34) : (i32) -> ()
# CHECK-NEXT:   %36 = func.call @gpt2(%32,%33) : (tensor<16xi32>,tensor<16xi32>) -> (tensor<*xf32>)
# CHECK-NEXT: }) : () -> ()
