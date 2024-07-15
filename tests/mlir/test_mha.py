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

def linear(x, w, b):
    return x @ w + b

def mha(x, n_head):
    bs = 16
    n_embed = 768
    n_embed_x3 = 2304
    n_head = 12
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
    out_heads = [head0, head1]
    x = linear(np.hstack(out_heads), c_attn_proj_w, c_attn_proj_b)
    return x

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
# CHECK-NEXT:   func.func @linear(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %24 = "tosa.matmul" (%arg0,%arg1) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %25 = "tosa.add" (%24,%arg2) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     return %25: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT:   func.func @mha(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> (tensor<*xf32>) {
# CHECK-NEXT:     %27 = "arith.constant" () {value = 16: i32} : () -> i32
# CHECK-NEXT:     %28 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:     %29 = "arith.constant" () {value = 2304: i32} : () -> i32
# CHECK-NEXT:     %30 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:     %31 = "numpy.random.randn" (%28,%29) : (i32,i32) -> tensor<768x2304xf32>
# CHECK-NEXT:     %32 = "numpy.random.randn" (%29) : (i32) -> tensor<2304xf32>
# CHECK-NEXT:     %33 = "numpy.random.randn" (%28,%28) : (i32,i32) -> tensor<768x768xf32>
# CHECK-NEXT:     %34 = "numpy.random.randn" (%28) : (i32) -> tensor<768xf32>
# CHECK-NEXT:     %35 = func.call @linear(%arg0,%31,%32) : (tensor<*xf32>,tensor<768x2304xf32>,tensor<2304xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     %36 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:     %37 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %38 = "numpy.split" (%35,%36,%37) {axis = array<i32: -1>} : (tensor<*xf32>,i32,i32) -> tensor<*xf32>
# CHECK-NEXT:     %39 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %40 = "tosa.gather" (%38,%39) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %41 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %42 = "numpy.split" (%40,%30,%41) {axis = array<i32: -1>} : (tensor<*xf32>,i32,i32) -> tensor<*xf32>
# CHECK-NEXT:     %43 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %44 = "tosa.gather" (%38,%43) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %45 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %46 = "numpy.split" (%44,%30,%45) {axis = array<i32: -1>} : (tensor<*xf32>,i32,i32) -> tensor<*xf32>
# CHECK-NEXT:     %47 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:     %48 = "tosa.gather" (%38,%47) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %49 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:     %50 = "numpy.split" (%48,%30,%49) {axis = array<i32: -1>} : (tensor<*xf32>,i32,i32) -> tensor<*xf32>
# CHECK-NEXT:     %51 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %52 = "numpy.tri" (%27) : (i32) -> i32
# CHECK-NEXT:     %53 = "arith.subi" (%51,%52) : (i32,i32) -> i32
# CHECK-NEXT:     %54 = "arith.constant" () {value = -1e-10: f32} : () -> f32
# CHECK-NEXT:     %55 = "arith.muli" (%53,%54) : (i32,f32) -> f32
# CHECK-NEXT:     %56 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %57 = "tosa.gather" (%42,%56) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %58 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %59 = "tosa.gather" (%46,%58) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %60 = "arith.constant" () {value = 0: i32} : () -> i32
# CHECK-NEXT:     %61 = "tosa.gather" (%50,%60) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %62 = func.call @attention(%57,%59,%61,%55) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %63 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %64 = "tosa.gather" (%42,%63) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %65 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %66 = "tosa.gather" (%46,%65) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %67 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:     %68 = "tosa.gather" (%50,%67) : (tensor<*xf32>,i32) -> tensor<*xf32>
# CHECK-NEXT:     %69 = func.call @attention(%64,%66,%68,%55) : (tensor<*xf32>,tensor<*xf32>,tensor<*xf32>,f32) -> (tensor<*xf32>)
# CHECK-NEXT:     %70 = "numpy.hstack" (%62,%69) : (tensor<*xf32>,tensor<*xf32>) -> tensor<*xf32>
# CHECK-NEXT:     %71 = func.call @linear(%70,%33,%34) : (tensor<*xf32>,tensor<768x768xf32>,tensor<768xf32>) -> (tensor<*xf32>)
# CHECK-NEXT:     return %71: tensor<*xf32>
# CHECK-NEXT:   }
# CHECK-NEXT: }) : () -> ()
