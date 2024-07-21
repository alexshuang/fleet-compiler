# RUN: fleet_compiler_cli %s --emitMLIR --only-compile | %FileCheck %s

import numpy as np

def gelu(x):
    pi = 3.141592653589793
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x):
    eps = 0.00001
    gamma = 0.00001
    beta = 0.00001
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta

def attention(q, k, v, mask):
    n_embed = 768
    n_head = 12
    n_embed_per_head = n_embed / n_head
    kt = np.transpose(k)
    qkt = q @ kt
    scaled_qkt = softmax(qkt / np.sqrt(n_embed_per_head) + mask)
    return scaled_qkt @ v

def linear(x, w, b):
    return x @ w + b

def ffn(x):
    n_embed = 768
    n_embed_x4 = 3072
    c_fc_w = np.random.randn(n_embed, n_embed_x4)
    c_fc_b = np.random.randn(n_embed_x4)
    c_proj_w = np.random.randn(n_embed_x4, n_embed)
    c_proj_b = np.random.randn(n_embed)
    return linear(gelu(linear(x, c_fc_w, c_fc_b)), c_proj_w, c_proj_b)

def mha(x, n_head):
    bs = 16
    n_embed = 768
    n_embed_x3 = 2304
    c_attn_w = np.random.randn(n_embed, n_embed_x3)
    c_attn_b = np.random.randn(n_embed_x3)
    c_attn_proj_w = np.random.randn(n_embed, n_embed)
    c_attn_proj_b = np.random.randn(n_embed)

    x = linear(x, c_attn_w, c_attn_b)
    qkv = np.split(x, 3, axis=-1)
    q = np.split(qkv[0], n_head, axis=-1)
    k = np.split(qkv[1], n_head, axis=-1)
    v = np.split(qkv[2], n_head, axis=-1)
    casual_mask = (1 - np.tri(bs)) * -0.0000000001

    # Not support for-loop, unroll it
    head0 = attention(q[0], k[0], v[0], casual_mask)
    head1 = attention(q[1], k[1], v[1], casual_mask)
    head2 = attention(q[2], k[2], v[2], casual_mask)
    head3 = attention(q[3], k[3], v[3], casual_mask)
    head4 = attention(q[4], k[4], v[4], casual_mask)
    head5 = attention(q[5], k[5], v[5], casual_mask)
    head6 = attention(q[6], k[6], v[6], casual_mask)
    head7 = attention(q[7], k[7], v[7], casual_mask)
    head8 = attention(q[8], k[8], v[8], casual_mask)
    head9 = attention(q[9], k[9], v[9], casual_mask)
    head10 = attention(q[10], k[10], v[10], casual_mask)
    head11 = attention(q[11], k[11], v[11], casual_mask)
    out_heads = [head0, head1, head2, head3, head4, head5, head6, head7, head8, head9, head10, head11]
    x = linear(np.hstack(out_heads), c_attn_proj_w, c_attn_proj_b)
    return x

def transformer_block(x):  # [n_seq, n_embd] -> [n_seq, n_embd]
    n_head = 12
    # multi-head casual self attention
    x = x + mha(layer_norm(x), n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # position-wise feed forward network
    x = x + ffn(layer_norm(x))  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x

def gpt2(x, p_x):
    vocab_size = 50257
    n_positions = 1024
    n_embed = 768
    wte = np.random.randn(vocab_size, n_embed)
    wpe = np.random.randn(n_positions, n_embed)

    x = wte[x]
    x = x + wpe[p_x]

    # Not support for-loop, unroll it
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)

    return layer_norm(x) @ np.transpose(wte)


np.random.seed(42)
inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
pos_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = gpt2(inputs, pos_array)
print(y)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   func.func @gelu(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %0 = "arith.constant" () {value = 3.141592653589793: f32} : () -> f32
# CHECK-NEXT:     %1 = "arith.constant" () {value = 0.5: f32} : () -> f32
# CHECK-NEXT:     %2 = "tensor.splat" (%1) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %3 = "tosa.mul" (%2,%arg0) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %4 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %5 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %6 = "arith.divsi" (%5,%0) : (i32,f32) -> f32
# CHECK-NEXT:     %7 = "numpy.sqrt" (%6) : (f32) -> f32
# CHECK-NEXT:     %8 = "arith.constant" () {value = 0.044715: f32} : () -> f32
# CHECK-NEXT:     %9 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %10 = "tensor.splat" (%9) : (i32) -> tensor<*xf32>
# CHECK-NEXT:     %11 = "tosa.pow" (%arg0,%10) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %12 = "tensor.splat" (%8) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %13 = "tosa.mul" (%12,%11) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %14 = "tosa.add" (%arg0,%13) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %15 = "tensor.splat" (%7) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %16 = "tosa.mul" (%15,%14) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %17 = "numpy.tanh" (%16) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %18 = "tensor.splat" (%4) : (i32) -> tensor<*xf32>
# CHECK-NEXT:     %19 = "tosa.add" (%18,%17) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %20 = "tosa.mul" (%3,%19) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %20: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @softmax(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %22 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %23 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %24 = "numpy.max" (%arg0,%22,%23) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %25 = "tosa.sub" (%arg0,%24) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %26 = "numpy.exp" (%25) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %27 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %28 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %29 = "numpy.sum" (%26,%27,%28) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %30 = "tosa.reciprocal" (%29) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %31 = "tosa.mul" (%26,%30) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %31: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @layer_norm(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %33 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %34 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %35 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:     %36 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %37 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %38 = "numpy.mean" (%arg0,%36,%37) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %39 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %40 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:     %41 = "numpy.var" (%arg0,%39,%40) {axis = array<i32: -1>,keepdims = true} : (tensor<*xf32>,i32,i1) -> tensor<*xf32>
# CHECK-NEXT:     %42 = "tosa.sub" (%arg0,%38) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %43 = "tensor.splat" (%34) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %44 = "tosa.mul" (%43,%42) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %45 = "tensor.splat" (%33) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %46 = "tosa.add" (%41,%45) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %47 = "numpy.sqrt" (%46) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %48 = "tosa.reciprocal" (%47) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %49 = "tosa.mul" (%44,%48) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %50 = "tensor.splat" (%35) : (f32) -> tensor<*xf32>
# CHECK-NEXT:     %51 = "tosa.add" (%49,%50) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %51: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @attention(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %53 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %54 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:     %55 = "arith.divsi" (%53,%54) : (i32,i32) -> i32
# CHECK-NEXT:     %56 = "numpy.transpose" (%arg1) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %57 = "tosa.matmul" (%arg0,%56) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %58 = "numpy.sqrt" (%55) : (i32) -> i32
# CHECK-NEXT:     %59 = "tensor.splat" (%58) : (i32) -> tensor<*xf32>
# CHECK-NEXT:     %60 = "tosa.reciprocal" (%59) : (tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %61 = "tosa.mul" (%57,%60) {shift = 0: i32} : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %62 = "tosa.add" (%61,%arg3) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %63 = func.call @softmax(%62) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %64 = "tosa.matmul" (%63,%arg2) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %64: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @linear(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %66 = "tosa.matmul" (%arg0,%arg1) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %67 = "tosa.add" (%66,%arg2) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %67: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @ffn(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %69 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %70 = "arith.constant" () {value = 3072: i32} : () -> i32
# CHECK-NEXT:     %71 = "numpy.random.randn" (%69,%70) : (i32,i32) -> tensor<768x3072xf32>
# CHECK-NEXT:     %72 = "numpy.random.randn" (%70) : (i32) -> tensor<3072xf32>
# CHECK-NEXT:     %73 = "numpy.random.randn" (%70,%69) : (i32,i32) -> tensor<3072x768xf32>
# CHECK-NEXT:     %74 = "numpy.random.randn" (%69) : (i32) -> tensor<768xf32>
# CHECK-NEXT:     %75 = func.call @linear(%arg0,%71,%72) : (tensor<*xf32>,tensor<768x3072xf32>,tensor<3072xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %76 = func.call @gelu(%75) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %77 = func.call @linear(%76,%73,%74) : (tensor<*xf32>,tensor<3072x768xf32>,tensor<768xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     return %77: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @mha(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %79 = "arith.constant" () {value = 16: i32} : () -> i32
# CHECK-NEXT:     %80 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %81 = "arith.constant" () {value = 2304: i32} : () -> i32
# CHECK-NEXT:     %82 = "numpy.random.randn" (%80,%81) : (i32,i32) -> tensor<768x2304xf32>
# CHECK-NEXT:     %83 = "numpy.random.randn" (%81) : (i32) -> tensor<2304xf32>
# CHECK-NEXT:     %84 = "numpy.random.randn" (%80,%80) : (i32,i32) -> tensor<768x768xf32>
# CHECK-NEXT:     %85 = "numpy.random.randn" (%80) : (i32) -> tensor<768xf32>
# CHECK-NEXT:     %86 = func.call @linear(%arg0,%82,%83) : (tensor<*xf32>,tensor<768x2304xf32>,tensor<2304xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %87 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %88 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %89 = "numpy.split" (%86,%87,%88) {axis = array<i32: -1>} : (tensor<*xf32>,i32,i32) -> tensor<*xf32>
# CHECK-NEXT:     %90 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %91 = "tosa.gather" (%89,%90) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %92 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %93 = "numpy.split" (%91,%arg1,%92) {axis = array<i32: -1>} : (tensor<*xf32>,tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %94 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %95 = "tosa.gather" (%89,%94) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %96 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %97 = "numpy.split" (%95,%arg1,%96) {axis = array<i32: -1>} : (tensor<*xf32>,tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %98 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %99 = "tosa.gather" (%89,%98) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %100 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %101 = "numpy.split" (%99,%arg1,%100) {axis = array<i32: -1>} : (tensor<*xf32>,tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %102 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %103 = "numpy.tri" (%79) : (i32) -> i32
# CHECK-NEXT:     %104 = "arith.subi" (%102,%103) : (i32,i32) -> i32
# CHECK-NEXT:     %105 = "arith.constant" () {value = -1e-10: f32} : () -> f32
# CHECK-NEXT:     %106 = "arith.muli" (%104,%105) : (i32,f32) -> f32
# CHECK-NEXT:     %107 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %108 = "tosa.gather" (%93,%107) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %109 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %110 = "tosa.gather" (%97,%109) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %111 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %112 = "tosa.gather" (%101,%111) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %113 = func.call @attention(%108,%110,%112,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %114 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %115 = "tosa.gather" (%93,%114) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %116 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %117 = "tosa.gather" (%97,%116) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %118 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %119 = "tosa.gather" (%101,%118) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %120 = func.call @attention(%115,%117,%119,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %121 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %122 = "tosa.gather" (%93,%121) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %123 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %124 = "tosa.gather" (%97,%123) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %125 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %126 = "tosa.gather" (%101,%125) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %127 = func.call @attention(%122,%124,%126,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %128 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %129 = "tosa.gather" (%93,%128) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %130 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %131 = "tosa.gather" (%97,%130) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %132 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %133 = "tosa.gather" (%101,%132) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %134 = func.call @attention(%129,%131,%133,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %135 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:     %136 = "tosa.gather" (%93,%135) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %137 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:     %138 = "tosa.gather" (%97,%137) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %139 = "arith.constant" () {value = 4: i32} : () -> i32
# CHECK-NEXT:     %140 = "tosa.gather" (%101,%139) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %141 = func.call @attention(%136,%138,%140,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %142 = "arith.constant" () {value = 5: i32} : () -> i32
# CHECK-NEXT:     %143 = "tosa.gather" (%93,%142) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %144 = "arith.constant" () {value = 5: i32} : () -> i32
# CHECK-NEXT:     %145 = "tosa.gather" (%97,%144) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %146 = "arith.constant" () {value = 5: i32} : () -> i32
# CHECK-NEXT:     %147 = "tosa.gather" (%101,%146) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %148 = func.call @attention(%143,%145,%147,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %149 = "arith.constant" () {value = 6: i32} : () -> i32
# CHECK-NEXT:     %150 = "tosa.gather" (%93,%149) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %151 = "arith.constant" () {value = 6: i32} : () -> i32
# CHECK-NEXT:     %152 = "tosa.gather" (%97,%151) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %153 = "arith.constant" () {value = 6: i32} : () -> i32
# CHECK-NEXT:     %154 = "tosa.gather" (%101,%153) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %155 = func.call @attention(%150,%152,%154,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %156 = "arith.constant" () {value = 7: i32} : () -> i32
# CHECK-NEXT:     %157 = "tosa.gather" (%93,%156) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %158 = "arith.constant" () {value = 7: i32} : () -> i32
# CHECK-NEXT:     %159 = "tosa.gather" (%97,%158) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %160 = "arith.constant" () {value = 7: i32} : () -> i32
# CHECK-NEXT:     %161 = "tosa.gather" (%101,%160) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %162 = func.call @attention(%157,%159,%161,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %163 = "arith.constant" () {value = 8: i32} : () -> i32
# CHECK-NEXT:     %164 = "tosa.gather" (%93,%163) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %165 = "arith.constant" () {value = 8: i32} : () -> i32
# CHECK-NEXT:     %166 = "tosa.gather" (%97,%165) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %167 = "arith.constant" () {value = 8: i32} : () -> i32
# CHECK-NEXT:     %168 = "tosa.gather" (%101,%167) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %169 = func.call @attention(%164,%166,%168,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %170 = "arith.constant" () {value = 9: i32} : () -> i32
# CHECK-NEXT:     %171 = "tosa.gather" (%93,%170) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %172 = "arith.constant" () {value = 9: i32} : () -> i32
# CHECK-NEXT:     %173 = "tosa.gather" (%97,%172) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %174 = "arith.constant" () {value = 9: i32} : () -> i32
# CHECK-NEXT:     %175 = "tosa.gather" (%101,%174) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %176 = func.call @attention(%171,%173,%175,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %177 = "arith.constant" () {value = 10: i32} : () -> i32
# CHECK-NEXT:     %178 = "tosa.gather" (%93,%177) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %179 = "arith.constant" () {value = 10: i32} : () -> i32
# CHECK-NEXT:     %180 = "tosa.gather" (%97,%179) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %181 = "arith.constant" () {value = 10: i32} : () -> i32
# CHECK-NEXT:     %182 = "tosa.gather" (%101,%181) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %183 = func.call @attention(%178,%180,%182,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %184 = "arith.constant" () {value = 11: i32} : () -> i32
# CHECK-NEXT:     %185 = "tosa.gather" (%93,%184) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %186 = "arith.constant" () {value = 11: i32} : () -> i32
# CHECK-NEXT:     %187 = "tosa.gather" (%97,%186) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %188 = "arith.constant" () {value = 11: i32} : () -> i32
# CHECK-NEXT:     %189 = "tosa.gather" (%101,%188) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %190 = func.call @attention(%185,%187,%189,%106) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %191 = "numpy.hstack" (%113,%120,%127,%134,%141,%148,%155,%162,%169,%176,%183,%190) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %192 = func.call @linear(%191,%84,%85) : (tensor<*xf32>,tensor<768x768xf32>,tensor<768xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     return %192: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @transformer_block(%arg0: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %194 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:     %195 = func.call @layer_norm(%arg0) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %196 = func.call @mha(%195,%194) : (tensor<*xf32>,i32) -> (tensor<*xf32>)
# CHECK-NEXT:     %197 = "tosa.add" (%arg0,%196) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %198 = func.call @layer_norm(%197) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %199 = func.call @ffn(%198) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %200 = "tosa.add" (%197,%199) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %200: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @gpt2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %202 = "arith.constant" () {value = 50257: i32} : () -> i32
# CHECK-NEXT:     %203 = "arith.constant" () {value = 1024: i32} : () -> i32
# CHECK-NEXT:     %204 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %205 = "numpy.random.randn" (%202,%204) : (i32,i32) -> tensor<50257x768xf32>
# CHECK-NEXT:     %206 = "numpy.random.randn" (%203,%204) : (i32,i32) -> tensor<1024x768xf32>
# CHECK-NEXT:     %207 = "tosa.gather" (%205,%arg0) : (tensor<50257x768xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %208 = "tosa.gather" (%206,%arg1) : (tensor<1024x768xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %209 = "tosa.add" (%207,%208) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %210 = func.call @transformer_block(%209) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %211 = func.call @transformer_block(%210) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %212 = func.call @transformer_block(%211) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %213 = func.call @transformer_block(%212) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %214 = func.call @transformer_block(%213) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %215 = func.call @transformer_block(%214) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %216 = func.call @transformer_block(%215) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %217 = func.call @transformer_block(%216) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %218 = func.call @transformer_block(%217) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %219 = func.call @transformer_block(%218) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %220 = func.call @transformer_block(%219) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %221 = func.call @transformer_block(%220) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %222 = func.call @layer_norm(%221) : (tensor<*xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %223 = "numpy.transpose" (%205) : (tensor<50257x768xf32>) -> tensor<768x50257xf32>
# CHECK-NEXT:     %224 = "tosa.matmul" (%222,%223) : (tensor<*xf32>,tensor<768x50257xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %224: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   %226 = "arith.constant" () {value = 42: i32} : () -> i32
# CHECK-NEXT:   %227 = "numpy.random.seed" (%226) : (i32) -> ()
# CHECK-NEXT:   %228 = "arith.constant" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %229 = "arith.constant" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %230 = func.call @gpt2(%228,%229) : (tensor<16xi32>,tensor<16xi32>) -> (tensor<*xf32>)
# CHECK-NEXT:   "python.print" (%230) : (tensor<*xf32>) -> ()
# CHECK-NEXT: }) : () -> ()
