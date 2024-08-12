# RUN: fleet_compiler_cli %s --emitMLIR --only-compile --opt | %FileCheck %s

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


np.random.seed(42)
inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
pos_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
y = gpt2(inputs, pos_array)
print(y)

# CHECK: "builtin.module" () ({
# CHECK-NEXT:   %256 = "arith.constant" () {value = 42: i32} : () -> i32
# CHECK-NEXT:   %258 = "arith.constant" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %259 = "arith.constant" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %1759 = "arith.constant" () {value = 50257: i32} : () -> i32
# CHECK-NEXT:   %1760 = "arith.constant" () {value = 1024: i32} : () -> i32
# CHECK-NEXT:   %1761 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1762 = "numpy.random.randn" (%1759,%1761) : (i32,i32) -> tensor<50257x768xf32>
# CHECK-NEXT:   %1763 = "numpy.random.randn" (%1760,%1761) : (i32,i32) -> tensor<1024x768xf32>
# CHECK-NEXT:   %1765 = "tosa.gather" (%1762,%258) : (tensor<50257x768xf32>,tensor<16xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1767 = "tosa.gather" (%1763,%259) : (tensor<1024x768xf32>,tensor<16xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1768 = "tosa.add" (%1765,%1767) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1770 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1772 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %1773 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %1774 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %1775 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1776 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2394 = "tosa.reduce_sum" (%1768) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2395 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2396 = "tosa.cast" (%2395) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2397 = "tosa.reciprocal" (%2396) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2398 = "tosa.mul" (%2394,%2397) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %1778 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1779 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2412 = "tosa.reduce_sum" (%1768) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2413 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2414 = "tosa.cast" (%2413) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2415 = "tosa.reciprocal" (%2414) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2416 = "tosa.mul" (%2412,%2415) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2418 = "tosa.cast" (%2416) : (tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2419 = "tosa.sub" (%1768,%2418) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2420 = "tosa.mul" (%2419,%2419) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2421 = "tosa.reduce_sum" (%1768) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2422 = "tosa.mul" (%2421,%2415) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %1781 = "tosa.sub" (%1768,%2398) : (tensor<16x768xf32>,tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1782 = "tensor.splat" (%1773) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %1783 = "tosa.mul" (%1782,%1781) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1784 = "tensor.splat" (%1772) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %1785 = "tosa.add" (%2422,%1784) : (tensor<16x1xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2378 = "math.sqrt" (%1785) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1787 = "tosa.reciprocal" (%2378) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1788 = "tosa.mul" (%1783,%1787) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1789 = "tensor.splat" (%1774) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %1790 = "tosa.add" (%1788,%1789) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1793 = "arith.constant" () {value = 16: i32} : () -> i32
# CHECK-NEXT:   %1794 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1795 = "arith.constant" () {value = 2304: i32} : () -> i32
# CHECK-NEXT:   %1796 = "numpy.random.randn" (%1794,%1795) : (i32,i32) -> tensor<768x2304xf32>
# CHECK-NEXT:   %1797 = "numpy.random.randn" (%1795) : (i32) -> tensor<2304xf32>
# CHECK-NEXT:   %1798 = "numpy.random.randn" (%1794,%1794) : (i32,i32) -> tensor<768x768xf32>
# CHECK-NEXT:   %1799 = "numpy.random.randn" (%1794) : (i32) -> tensor<768xf32>
# CHECK-NEXT:   %1803 = "tosa.matmul" (%1790,%1796) : (tensor<16x768xf32>,tensor<768x2304xf32>) -> tensor<16x2304xf32>
# CHECK-NEXT:   %1804 = "tosa.add" (%1803,%1797) : (tensor<16x2304xf32>,tensor<2304xf32>) -> tensor<16x2304xf32>
# CHECK-NEXT:   %1805 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %1806 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1807 = "numpy.split" (%1804,%1805,%1806) {axis = [-1]} : (tensor<16x2304xf32>,i32,i32) -> tensor<3x16x768xf32>
# CHECK-NEXT:   %2339 = "tosa.const" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1810 = "tosa.gather" (%1807,%2339) : (tensor<3x16x768xf32>,tensor<1xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1811 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1812 = "numpy.split" (%1810,%1770,%1811) {axis = [-1]} : (tensor<16x768xf32>,i32,i32) -> tensor<12x16x64xf32>
# CHECK-NEXT:   %2340 = "tosa.const" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1815 = "tosa.gather" (%1807,%2340) : (tensor<3x16x768xf32>,tensor<1xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1816 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1817 = "numpy.split" (%1815,%1770,%1816) {axis = [-1]} : (tensor<16x768xf32>,i32,i32) -> tensor<12x16x64xf32>
# CHECK-NEXT:   %2341 = "tosa.const" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1820 = "tosa.gather" (%1807,%2341) : (tensor<3x16x768xf32>,tensor<1xi32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %1821 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1822 = "numpy.split" (%1820,%1770,%1821) {axis = [-1]} : (tensor<16x768xf32>,i32,i32) -> tensor<12x16x64xf32>
# CHECK-NEXT:   %1823 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:   %1824 = "numpy.tri" (%1793) : (i32) -> i32
# CHECK-NEXT:   %1825 = "arith.subi" (%1823,%1824) : (i32,i32) -> i32
# CHECK-NEXT:   %1826 = "arith.constant" () {value = -1e-10: f32} : () -> f32
# CHECK-NEXT:   %1827 = "arith.muli" (%1825,%1826) : (i32,f32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2342 = "tosa.const" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1830 = "tosa.gather" (%1812,%2342) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2343 = "tosa.const" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1833 = "tosa.gather" (%1817,%2343) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2344 = "tosa.const" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1836 = "tosa.gather" (%1822,%2344) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %1841 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1842 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1843 = "arith.divsi" (%1841,%1842) : (i32,i32) -> i32
# CHECK-NEXT:   %1844 = "numpy.transpose" (%1833) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %1845 = "tosa.matmul" (%1830,%1844) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2379 = "math.sqrt" (%1843) : (i32) -> i32
# CHECK-NEXT:   %1847 = "tensor.splat" (%2379) : (i32) -> i32
# CHECK-NEXT:   %1848 = "tosa.reciprocal" (%1847) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %1849 = "tosa.mul" (%1845,%1848) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1850 = "tosa.add" (%1849,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1852 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1853 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1854 = "numpy.max" (%1850,%1852,%1853) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1855 = "tosa.sub" (%1850,%1854) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1856 = "numpy.exp" (%1855) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1857 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1858 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1859 = "numpy.sum" (%1856,%1857,%1858) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1860 = "tosa.reciprocal" (%1859) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1861 = "tosa.mul" (%1856,%1860) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1862 = "tosa.matmul" (%1861,%1836) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2345 = "tosa.const" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1865 = "tosa.gather" (%1812,%2345) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2346 = "tosa.const" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1868 = "tosa.gather" (%1817,%2346) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2347 = "tosa.const" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1871 = "tosa.gather" (%1822,%2347) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %1876 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1877 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1878 = "arith.divsi" (%1876,%1877) : (i32,i32) -> i32
# CHECK-NEXT:   %1879 = "numpy.transpose" (%1868) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %1880 = "tosa.matmul" (%1865,%1879) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2380 = "math.sqrt" (%1878) : (i32) -> i32
# CHECK-NEXT:   %1882 = "tensor.splat" (%2380) : (i32) -> i32
# CHECK-NEXT:   %1883 = "tosa.reciprocal" (%1882) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %1884 = "tosa.mul" (%1880,%1883) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1885 = "tosa.add" (%1884,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1887 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1888 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1889 = "numpy.max" (%1885,%1887,%1888) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1890 = "tosa.sub" (%1885,%1889) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1891 = "numpy.exp" (%1890) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1892 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1893 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1894 = "numpy.sum" (%1891,%1892,%1893) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1895 = "tosa.reciprocal" (%1894) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1896 = "tosa.mul" (%1891,%1895) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1897 = "tosa.matmul" (%1896,%1871) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2348 = "tosa.const" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1900 = "tosa.gather" (%1812,%2348) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2349 = "tosa.const" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1903 = "tosa.gather" (%1817,%2349) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2350 = "tosa.const" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1906 = "tosa.gather" (%1822,%2350) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %1911 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1912 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1913 = "arith.divsi" (%1911,%1912) : (i32,i32) -> i32
# CHECK-NEXT:   %1914 = "numpy.transpose" (%1903) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %1915 = "tosa.matmul" (%1900,%1914) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2381 = "math.sqrt" (%1913) : (i32) -> i32
# CHECK-NEXT:   %1917 = "tensor.splat" (%2381) : (i32) -> i32
# CHECK-NEXT:   %1918 = "tosa.reciprocal" (%1917) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %1919 = "tosa.mul" (%1915,%1918) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1920 = "tosa.add" (%1919,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1922 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1923 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1924 = "numpy.max" (%1920,%1922,%1923) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1925 = "tosa.sub" (%1920,%1924) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1926 = "numpy.exp" (%1925) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1927 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1928 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1929 = "numpy.sum" (%1926,%1927,%1928) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1930 = "tosa.reciprocal" (%1929) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1931 = "tosa.mul" (%1926,%1930) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1932 = "tosa.matmul" (%1931,%1906) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2351 = "tosa.const" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1935 = "tosa.gather" (%1812,%2351) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2352 = "tosa.const" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1938 = "tosa.gather" (%1817,%2352) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2353 = "tosa.const" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1941 = "tosa.gather" (%1822,%2353) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %1946 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1947 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1948 = "arith.divsi" (%1946,%1947) : (i32,i32) -> i32
# CHECK-NEXT:   %1949 = "numpy.transpose" (%1938) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %1950 = "tosa.matmul" (%1935,%1949) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2382 = "math.sqrt" (%1948) : (i32) -> i32
# CHECK-NEXT:   %1952 = "tensor.splat" (%2382) : (i32) -> i32
# CHECK-NEXT:   %1953 = "tosa.reciprocal" (%1952) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %1954 = "tosa.mul" (%1950,%1953) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1955 = "tosa.add" (%1954,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1957 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1958 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1959 = "numpy.max" (%1955,%1957,%1958) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1960 = "tosa.sub" (%1955,%1959) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1961 = "numpy.exp" (%1960) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1962 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1963 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1964 = "numpy.sum" (%1961,%1962,%1963) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1965 = "tosa.reciprocal" (%1964) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1966 = "tosa.mul" (%1961,%1965) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1967 = "tosa.matmul" (%1966,%1941) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2354 = "tosa.const" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1970 = "tosa.gather" (%1812,%2354) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2355 = "tosa.const" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1973 = "tosa.gather" (%1817,%2355) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2356 = "tosa.const" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %1976 = "tosa.gather" (%1822,%2356) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %1981 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %1982 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %1983 = "arith.divsi" (%1981,%1982) : (i32,i32) -> i32
# CHECK-NEXT:   %1984 = "numpy.transpose" (%1973) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %1985 = "tosa.matmul" (%1970,%1984) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2383 = "math.sqrt" (%1983) : (i32) -> i32
# CHECK-NEXT:   %1987 = "tensor.splat" (%2383) : (i32) -> i32
# CHECK-NEXT:   %1988 = "tosa.reciprocal" (%1987) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %1989 = "tosa.mul" (%1985,%1988) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1990 = "tosa.add" (%1989,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1992 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1993 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1994 = "numpy.max" (%1990,%1992,%1993) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %1995 = "tosa.sub" (%1990,%1994) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1996 = "numpy.exp" (%1995) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %1997 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %1998 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %1999 = "numpy.sum" (%1996,%1997,%1998) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2000 = "tosa.reciprocal" (%1999) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2001 = "tosa.mul" (%1996,%2000) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2002 = "tosa.matmul" (%2001,%1976) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2357 = "tosa.const" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2005 = "tosa.gather" (%1812,%2357) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2358 = "tosa.const" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2008 = "tosa.gather" (%1817,%2358) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2359 = "tosa.const" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2011 = "tosa.gather" (%1822,%2359) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2016 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2017 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2018 = "arith.divsi" (%2016,%2017) : (i32,i32) -> i32
# CHECK-NEXT:   %2019 = "numpy.transpose" (%2008) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2020 = "tosa.matmul" (%2005,%2019) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2384 = "math.sqrt" (%2018) : (i32) -> i32
# CHECK-NEXT:   %2022 = "tensor.splat" (%2384) : (i32) -> i32
# CHECK-NEXT:   %2023 = "tosa.reciprocal" (%2022) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2024 = "tosa.mul" (%2020,%2023) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2025 = "tosa.add" (%2024,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2027 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2028 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2029 = "numpy.max" (%2025,%2027,%2028) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2030 = "tosa.sub" (%2025,%2029) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2031 = "numpy.exp" (%2030) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2032 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2033 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2034 = "numpy.sum" (%2031,%2032,%2033) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2035 = "tosa.reciprocal" (%2034) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2036 = "tosa.mul" (%2031,%2035) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2037 = "tosa.matmul" (%2036,%2011) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2360 = "tosa.const" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2040 = "tosa.gather" (%1812,%2360) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2361 = "tosa.const" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2043 = "tosa.gather" (%1817,%2361) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2362 = "tosa.const" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2046 = "tosa.gather" (%1822,%2362) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2051 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2052 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2053 = "arith.divsi" (%2051,%2052) : (i32,i32) -> i32
# CHECK-NEXT:   %2054 = "numpy.transpose" (%2043) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2055 = "tosa.matmul" (%2040,%2054) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2385 = "math.sqrt" (%2053) : (i32) -> i32
# CHECK-NEXT:   %2057 = "tensor.splat" (%2385) : (i32) -> i32
# CHECK-NEXT:   %2058 = "tosa.reciprocal" (%2057) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2059 = "tosa.mul" (%2055,%2058) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2060 = "tosa.add" (%2059,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2062 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2063 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2064 = "numpy.max" (%2060,%2062,%2063) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2065 = "tosa.sub" (%2060,%2064) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2066 = "numpy.exp" (%2065) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2067 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2068 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2069 = "numpy.sum" (%2066,%2067,%2068) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2070 = "tosa.reciprocal" (%2069) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2071 = "tosa.mul" (%2066,%2070) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2072 = "tosa.matmul" (%2071,%2046) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2363 = "tosa.const" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2075 = "tosa.gather" (%1812,%2363) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2364 = "tosa.const" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2078 = "tosa.gather" (%1817,%2364) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2365 = "tosa.const" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2081 = "tosa.gather" (%1822,%2365) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2086 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2087 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2088 = "arith.divsi" (%2086,%2087) : (i32,i32) -> i32
# CHECK-NEXT:   %2089 = "numpy.transpose" (%2078) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2090 = "tosa.matmul" (%2075,%2089) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2386 = "math.sqrt" (%2088) : (i32) -> i32
# CHECK-NEXT:   %2092 = "tensor.splat" (%2386) : (i32) -> i32
# CHECK-NEXT:   %2093 = "tosa.reciprocal" (%2092) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2094 = "tosa.mul" (%2090,%2093) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2095 = "tosa.add" (%2094,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2097 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2098 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2099 = "numpy.max" (%2095,%2097,%2098) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2100 = "tosa.sub" (%2095,%2099) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2101 = "numpy.exp" (%2100) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2102 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2103 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2104 = "numpy.sum" (%2101,%2102,%2103) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2105 = "tosa.reciprocal" (%2104) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2106 = "tosa.mul" (%2101,%2105) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2107 = "tosa.matmul" (%2106,%2081) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2366 = "tosa.const" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2110 = "tosa.gather" (%1812,%2366) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2367 = "tosa.const" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2113 = "tosa.gather" (%1817,%2367) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2368 = "tosa.const" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2116 = "tosa.gather" (%1822,%2368) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2121 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2122 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2123 = "arith.divsi" (%2121,%2122) : (i32,i32) -> i32
# CHECK-NEXT:   %2124 = "numpy.transpose" (%2113) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2125 = "tosa.matmul" (%2110,%2124) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2387 = "math.sqrt" (%2123) : (i32) -> i32
# CHECK-NEXT:   %2127 = "tensor.splat" (%2387) : (i32) -> i32
# CHECK-NEXT:   %2128 = "tosa.reciprocal" (%2127) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2129 = "tosa.mul" (%2125,%2128) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2130 = "tosa.add" (%2129,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2132 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2133 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2134 = "numpy.max" (%2130,%2132,%2133) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2135 = "tosa.sub" (%2130,%2134) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2136 = "numpy.exp" (%2135) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2137 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2138 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2139 = "numpy.sum" (%2136,%2137,%2138) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2140 = "tosa.reciprocal" (%2139) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2141 = "tosa.mul" (%2136,%2140) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2142 = "tosa.matmul" (%2141,%2116) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2369 = "tosa.const" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2145 = "tosa.gather" (%1812,%2369) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2370 = "tosa.const" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2148 = "tosa.gather" (%1817,%2370) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2371 = "tosa.const" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2151 = "tosa.gather" (%1822,%2371) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2156 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2157 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2158 = "arith.divsi" (%2156,%2157) : (i32,i32) -> i32
# CHECK-NEXT:   %2159 = "numpy.transpose" (%2148) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2160 = "tosa.matmul" (%2145,%2159) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2388 = "math.sqrt" (%2158) : (i32) -> i32
# CHECK-NEXT:   %2162 = "tensor.splat" (%2388) : (i32) -> i32
# CHECK-NEXT:   %2163 = "tosa.reciprocal" (%2162) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2164 = "tosa.mul" (%2160,%2163) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2165 = "tosa.add" (%2164,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2167 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2168 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2169 = "numpy.max" (%2165,%2167,%2168) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2170 = "tosa.sub" (%2165,%2169) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2171 = "numpy.exp" (%2170) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2172 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2173 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2174 = "numpy.sum" (%2171,%2172,%2173) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2175 = "tosa.reciprocal" (%2174) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2176 = "tosa.mul" (%2171,%2175) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2177 = "tosa.matmul" (%2176,%2151) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2372 = "tosa.const" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2180 = "tosa.gather" (%1812,%2372) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2373 = "tosa.const" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2183 = "tosa.gather" (%1817,%2373) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2374 = "tosa.const" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2186 = "tosa.gather" (%1822,%2374) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2191 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2192 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2193 = "arith.divsi" (%2191,%2192) : (i32,i32) -> i32
# CHECK-NEXT:   %2194 = "numpy.transpose" (%2183) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2195 = "tosa.matmul" (%2180,%2194) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2389 = "math.sqrt" (%2193) : (i32) -> i32
# CHECK-NEXT:   %2197 = "tensor.splat" (%2389) : (i32) -> i32
# CHECK-NEXT:   %2198 = "tosa.reciprocal" (%2197) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2199 = "tosa.mul" (%2195,%2198) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2200 = "tosa.add" (%2199,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2202 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2203 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2204 = "numpy.max" (%2200,%2202,%2203) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2205 = "tosa.sub" (%2200,%2204) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2206 = "numpy.exp" (%2205) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2207 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2208 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2209 = "numpy.sum" (%2206,%2207,%2208) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2210 = "tosa.reciprocal" (%2209) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2211 = "tosa.mul" (%2206,%2210) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2212 = "tosa.matmul" (%2211,%2186) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2375 = "tosa.const" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2215 = "tosa.gather" (%1812,%2375) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2376 = "tosa.const" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2218 = "tosa.gather" (%1817,%2376) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2377 = "tosa.const" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2221 = "tosa.gather" (%1822,%2377) : (tensor<12x16x64xf32>,tensor<1xi32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2226 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2227 = "arith.constant" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2228 = "arith.divsi" (%2226,%2227) : (i32,i32) -> i32
# CHECK-NEXT:   %2229 = "numpy.transpose" (%2218) {axes = none} : (tensor<16x64xf32>) -> tensor<64x16xf32>
# CHECK-NEXT:   %2230 = "tosa.matmul" (%2215,%2229) : (tensor<16x64xf32>,tensor<64x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2390 = "math.sqrt" (%2228) : (i32) -> i32
# CHECK-NEXT:   %2232 = "tensor.splat" (%2390) : (i32) -> i32
# CHECK-NEXT:   %2233 = "tosa.reciprocal" (%2232) : (i32) -> tensor<16x16xf32>
# CHECK-NEXT:   %2234 = "tosa.mul" (%2230,%2233) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2235 = "tosa.add" (%2234,%1827) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2237 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2238 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2239 = "numpy.max" (%2235,%2237,%2238) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2240 = "tosa.sub" (%2235,%2239) : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2241 = "numpy.exp" (%2240) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2242 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2243 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2244 = "numpy.sum" (%2241,%2242,%2243) {axis = [-1],keepdims = true} : (tensor<16x16xf32>,i32,i1) -> tensor<16x16xf32>
# CHECK-NEXT:   %2245 = "tosa.reciprocal" (%2244) : (tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2246 = "tosa.mul" (%2241,%2245) {shift = 0: i32} : (tensor<16x16xf32>,tensor<16x16xf32>) -> tensor<16x16xf32>
# CHECK-NEXT:   %2247 = "tosa.matmul" (%2246,%2221) : (tensor<16x16xf32>,tensor<16x64xf32>) -> tensor<16x64xf32>
# CHECK-NEXT:   %2248 = "numpy.hstack" (%1862,%1897,%1932,%1967,%2002,%2037,%2072,%2107,%2142,%2177,%2212,%2247) : (tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2252 = "tosa.matmul" (%2248,%1798) : (tensor<16x768xf32>,tensor<768x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2253 = "tosa.add" (%2252,%1799) : (tensor<16x768xf32>,tensor<768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2254 = "tosa.add" (%1768,%2253) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2256 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2257 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2258 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2259 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2260 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2400 = "tosa.reduce_sum" (%2254) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2401 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2402 = "tosa.cast" (%2401) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2403 = "tosa.reciprocal" (%2402) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2404 = "tosa.mul" (%2400,%2403) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2262 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2263 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2424 = "tosa.reduce_sum" (%2254) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2425 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2426 = "tosa.cast" (%2425) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2427 = "tosa.reciprocal" (%2426) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2428 = "tosa.mul" (%2424,%2427) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2430 = "tosa.cast" (%2428) : (tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2431 = "tosa.sub" (%2254,%2430) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2432 = "tosa.mul" (%2431,%2431) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2433 = "tosa.reduce_sum" (%2254) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2434 = "tosa.mul" (%2433,%2427) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2265 = "tosa.sub" (%2254,%2404) : (tensor<16x768xf32>,tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2266 = "tensor.splat" (%2257) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2267 = "tosa.mul" (%2266,%2265) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2268 = "tensor.splat" (%2256) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2269 = "tosa.add" (%2434,%2268) : (tensor<16x1xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2391 = "math.sqrt" (%2269) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2271 = "tosa.reciprocal" (%2391) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2272 = "tosa.mul" (%2267,%2271) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2273 = "tensor.splat" (%2258) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2274 = "tosa.add" (%2272,%2273) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2276 = "arith.constant" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2277 = "arith.constant" () {value = 3072: i32} : () -> i32
# CHECK-NEXT:   %2278 = "numpy.random.randn" (%2276,%2277) : (i32,i32) -> tensor<768x3072xf32>
# CHECK-NEXT:   %2279 = "numpy.random.randn" (%2277) : (i32) -> tensor<3072xf32>
# CHECK-NEXT:   %2280 = "numpy.random.randn" (%2277,%2276) : (i32,i32) -> tensor<3072x768xf32>
# CHECK-NEXT:   %2281 = "numpy.random.randn" (%2276) : (i32) -> tensor<768xf32>
# CHECK-NEXT:   %2285 = "tosa.matmul" (%2274,%2278) : (tensor<16x768xf32>,tensor<768x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2286 = "tosa.add" (%2285,%2279) : (tensor<16x3072xf32>,tensor<3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2288 = "arith.constant" () {value = 3.141592653589793: f32} : () -> f32
# CHECK-NEXT:   %2289 = "arith.constant" () {value = 0.5: f32} : () -> f32
# CHECK-NEXT:   %2290 = "tensor.splat" (%2289) : (f32) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2291 = "tosa.mul" (%2290,%2286) {shift = 0: i32} : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2292 = "arith.constant" () {value = 1: i32} : () -> i32
# CHECK-NEXT:   %2293 = "arith.constant" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %2294 = "arith.divsi" (%2293,%2288) : (i32,f32) -> f32
# CHECK-NEXT:   %2392 = "math.sqrt" (%2294) : (f32) -> f32
# CHECK-NEXT:   %2296 = "arith.constant" () {value = 0.044715: f32} : () -> f32
# CHECK-NEXT:   %2297 = "arith.constant" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %2298 = "tensor.splat" (%2297) : (i32) -> i32
# CHECK-NEXT:   %2299 = "tosa.pow" (%2286,%2298) : (tensor<16x3072xf32>,i32) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2300 = "tensor.splat" (%2296) : (f32) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2301 = "tosa.mul" (%2300,%2299) {shift = 0: i32} : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2302 = "tosa.add" (%2286,%2301) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2303 = "tensor.splat" (%2392) : (f32) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2304 = "tosa.mul" (%2303,%2302) {shift = 0: i32} : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2305 = "numpy.tanh" (%2304) : (tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2306 = "tensor.splat" (%2292) : (i32) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2307 = "tosa.add" (%2306,%2305) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2308 = "tosa.mul" (%2291,%2307) {shift = 0: i32} : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> tensor<16x3072xf32>
# CHECK-NEXT:   %2312 = "tosa.matmul" (%2308,%2280) : (tensor<16x3072xf32>,tensor<3072x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2313 = "tosa.add" (%2312,%2281) : (tensor<16x768xf32>,tensor<768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2314 = "tosa.add" (%2254,%2313) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2316 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2317 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2318 = "arith.constant" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2319 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2320 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2406 = "tosa.reduce_sum" (%2314) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2407 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2408 = "tosa.cast" (%2407) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2409 = "tosa.reciprocal" (%2408) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2410 = "tosa.mul" (%2406,%2409) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2322 = "arith.constant" () {value = -1: i32} : () -> i32
# CHECK-NEXT:   %2323 = "arith.constant" () {value = true} : () -> i1
# CHECK-NEXT:   %2436 = "tosa.reduce_sum" (%2314) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2437 = "tosa.const" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2438 = "tosa.cast" (%2437) : (tensor<1xi32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2439 = "tosa.reciprocal" (%2438) : (tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2440 = "tosa.mul" (%2436,%2439) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2442 = "tosa.cast" (%2440) : (tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2443 = "tosa.sub" (%2314,%2442) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2444 = "tosa.mul" (%2443,%2443) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2445 = "tosa.reduce_sum" (%2314) {axis = -1: i64} : (tensor<16x768xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2446 = "tosa.mul" (%2445,%2439) {shift = 0: i32} : (tensor<16x1xf32>,tensor<16x1xf32>) -> tensor<16x1xf32>
# CHECK-NEXT:   %2325 = "tosa.sub" (%2314,%2410) : (tensor<16x768xf32>,tensor<16x1xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2326 = "tensor.splat" (%2317) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2327 = "tosa.mul" (%2326,%2325) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2328 = "tensor.splat" (%2316) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2329 = "tosa.add" (%2446,%2328) : (tensor<16x1xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2393 = "math.sqrt" (%2329) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2331 = "tosa.reciprocal" (%2393) : (tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2332 = "tosa.mul" (%2327,%2331) {shift = 0: i32} : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2333 = "tensor.splat" (%2318) : (f32) -> tensor<16x768xf32>
# CHECK-NEXT:   %2334 = "tosa.add" (%2332,%2333) : (tensor<16x768xf32>,tensor<16x768xf32>) -> tensor<16x768xf32>
# CHECK-NEXT:   %2335 = "numpy.transpose" (%1762) {axes = [1, 0]} : (tensor<50257x768xf32>) -> tensor<768x50257xf32>
# CHECK-NEXT:   %2336 = "tosa.matmul" (%2334,%2335) : (tensor<16x768xf32>,tensor<768x50257xf32>) -> tensor<16x50257xf32>
# CHECK-NEXT:   "python.print" (%2336) : (tensor<16x50257xf32>) -> ()
# CHECK-NEXT: }) : () -> ()
