# RUN: fleet_compiler_cli %s --emitMLIR --only-compile --opt --target-backend sycl | %FileCheck %s

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
# CHECK-NEXT:   %2977 = "vm.rodata" () {value = dense<[8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %2978 = "vm.rodata" () {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]>: tensor<16xi32>} : () -> tensor<16xi32>
# CHECK-NEXT:   %2979 = "vm.rodata" () {value = dense<[ 0.49671415 -0.1382643   0.64768854 ... -0.44119019 -0.40071032
# CHECK-NEXT:  -0.52496616]>: tensor<50257x768xf32>} : () -> tensor<50257x768xf32>
# CHECK-NEXT:   %2980 = "vm.rodata" () {value = dense<[ 0.8408791  -1.97249614 -0.62756831 ...  0.95089149 -2.54648594
# CHECK-NEXT:  -0.40704988]>: tensor<1024x768xf32>} : () -> tensor<1024x768xf32>
# CHECK-NEXT:   %2893 = "vm.call" @device.gather(%2979,%2977) : (tensor<50257x768xf32>,tensor<16xi32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2894 = "vm.call" @device.gather(%2980,%2978) : (tensor<1024x768xf32>,tensor<16xi32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2769 = "vm.call" @device.add(%2893,%2894) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2572 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2573 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2574 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2935 = "vm.call" @device.reduce_sum(%2769) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2981 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2968 = "vm.call" @device.cast(%2981) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3118 = "vm.call" @device.div(%2935,%2968) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2936 = "vm.call" @device.reduce_sum(%2769) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2982 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2969 = "vm.call" @device.cast(%2982) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3120 = "vm.call" @device.div(%2936,%2969) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2970 = "vm.call" @device.cast(%3120) : (tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2796 = "vm.call" @device.sub(%2769,%2970) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2816 = "vm.call" @device.mul(%2796,%2796) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2937 = "vm.call" @device.reduce_sum(%2769) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3119 = "vm.call" @device.div(%2937,%2969) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2797 = "vm.call" @device.sub(%2769,%3118) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2698 = "vm.call" @device.splat(%2573) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2818 = "vm.call" @device.mul(%2698,%2797) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2699 = "vm.call" @device.splat(%2572) : (f32) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2770 = "vm.call" @device.add(%3119,%2699) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2724 = "vm.call" @device.sqrt(%2770) : (tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3121 = "vm.call" @device.div(%2818,%2724) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2700 = "vm.call" @device.splat(%2574) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2771 = "vm.call" @device.add(%3121,%2700) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2983 = "vm.rodata" () {value = dense<[0.58451456 0.68425926 1.38721336 ... 1.50509363 0.53584878 0.91657596]>: tensor<768x2304xf32>} : () -> tensor<768x2304xf32>
# CHECK-NEXT:   %2984 = "vm.rodata" () {value = dense<[-0.8546148   1.06789871  0.79961198 ... -2.41904842  0.17527923
# CHECK-NEXT:  -0.1904177 ]>: tensor<2304xf32>} : () -> tensor<2304xf32>
# CHECK-NEXT:   %2985 = "vm.rodata" () {value = dense<[-1.12767487  0.02913428 -1.56297829 ... -1.81213502 -0.93844992
# CHECK-NEXT:  -0.18654629]>: tensor<768x768xf32>} : () -> tensor<768x768xf32>
# CHECK-NEXT:   %2986 = "vm.rodata" () {value = dense<[-1.10943658e+00 -1.19798016e-01  5.96037876e-01  1.19580427e-01
# CHECK-NEXT:  -1.53857411e+00  7.81947239e-02  1.96630925e+00 -8.08554297e-01
# CHECK-NEXT:   9.12124983e-01  1.60482917e+00 -8.97196416e-01 -1.12109589e-01
# CHECK-NEXT:   5.01878775e-01  4.60917962e-01 -6.14605375e-01  6.03570171e-02
# CHECK-NEXT:   2.35144139e-01  1.53426730e-01 -1.51231556e-01 -9.98495011e-01
# CHECK-NEXT:  -1.50694469e+00  9.46282845e-02 -1.10588489e+00 -4.94183345e-01
# CHECK-NEXT:  -1.12581376e+00 -5.37368562e-01 -4.58804561e-01  1.59502540e+00
# CHECK-NEXT:   8.81012486e-01 -5.91387851e-01 -4.17660081e-02 -9.78806407e-01
# CHECK-NEXT:   1.07687345e+00  3.50489436e-01  1.03048595e+00  3.56577828e-01
# CHECK-NEXT:   1.09264527e+00 -1.26501319e+00 -8.88623169e-01 -2.37334985e-01
# CHECK-NEXT:   9.76814421e-01 -1.34895071e-01  6.41866811e-01 -1.47879451e+00
# CHECK-NEXT:  -4.68788580e-02 -6.29065727e-01 -5.34727958e-01  7.59142239e-02
# CHECK-NEXT:  -1.88761001e-01 -8.03285349e-01 -8.31774742e-01  2.59541589e-02
# CHECK-NEXT:   3.06868696e-01  1.08445106e+00 -8.48500921e-01  5.63660997e-02
# CHECK-NEXT:   2.14013201e-01  1.23526771e+00 -2.97915643e-01  8.49010304e-01
# CHECK-NEXT:   5.03942797e-01 -7.25134105e-01 -1.57582004e+00  5.44618757e-01
# CHECK-NEXT:   3.98514311e-01 -1.20903068e+00 -3.02917961e-01  6.56514085e-01
# CHECK-NEXT:   1.43602461e-01  3.57023457e-01 -3.26485406e-01  2.55763342e-01
# CHECK-NEXT:  -1.48346412e+00  1.16236803e+00  2.83224573e-01  3.46916972e-01
# CHECK-NEXT:   7.61982622e-02 -1.34219313e-01 -7.51740377e-01 -3.74930937e-01
# CHECK-NEXT:   3.74746903e-01  1.22574354e+00  1.28957760e+00 -6.41571139e-01
# CHECK-NEXT:  -5.39786685e-01  6.19858122e-02 -5.62737293e-01  7.85363984e-01
# CHECK-NEXT:  -4.47153168e-01 -6.86067765e-02 -5.88196186e-02 -5.59723079e-01
# CHECK-NEXT:   3.56851935e-01  1.29334232e+00  3.77912676e-01 -1.28794194e+00
# CHECK-NEXT:  -1.21992638e+00  1.11217016e+00  3.36848990e-01 -6.72295564e-01
# CHECK-NEXT:   9.79693181e-01  1.25255378e+00 -1.58359639e-01  4.55058936e-01
# CHECK-NEXT:   9.27234089e-01  6.18796626e-01 -1.47083200e-02 -1.98597393e-02
# CHECK-NEXT:  -6.09765614e-01  1.17021966e+00  3.45254027e-03 -7.36373294e-01
# CHECK-NEXT:   1.37287117e-01 -5.57942541e-01 -4.00636387e-01 -1.20496722e+00
# CHECK-NEXT:  -1.66398677e-01 -3.31700400e-01  3.79382759e-02 -9.38213130e-01
# CHECK-NEXT:   5.30873039e-01  1.36917892e+00  4.12132855e-01  1.06378251e-01
# CHECK-NEXT:   1.40284224e+00 -2.10359049e+00  5.73911099e-02 -6.64639667e-01
# CHECK-NEXT:  -1.70278004e+00  1.32358077e-01  1.97599028e+00 -1.84397732e+00
# CHECK-NEXT:  -8.13232401e-02  9.62385445e-01  1.87030516e+00 -8.98043658e-02
# CHECK-NEXT:  -1.87549447e+00 -6.96802040e-01  9.83922679e-01  3.99314267e-02
# CHECK-NEXT:  -7.59377642e-02  4.01091275e-01 -6.94753581e-01  1.55567208e-01
# CHECK-NEXT:  -1.39338405e+00  9.99942091e-01  6.15567556e-01 -1.00444776e+00
# CHECK-NEXT:  -1.42696677e-01 -1.06243135e+00  2.36674190e-01 -1.76148485e+00
# CHECK-NEXT:   1.94253731e+00  2.87679725e-01  2.17348623e-01  2.67567006e-01
# CHECK-NEXT:  -8.19371754e-01  5.00005826e-01  2.53008539e-01  1.07337921e-01
# CHECK-NEXT:  -6.23231038e-01  1.95741683e-01  6.97088528e-01  1.29121778e+00
# CHECK-NEXT:   1.04330895e+00  6.13131450e-02 -6.56640632e-01  4.09161894e-01
# CHECK-NEXT:  -2.57475665e-01 -8.27843913e-01  1.22000029e-01 -5.68562736e-02
# CHECK-NEXT:   2.19600873e-01 -2.43226186e-01  7.87453576e-01  2.49554570e-01
# CHECK-NEXT:  -7.44261640e-01  5.79051524e-01  1.47307914e+00  1.21032848e+00
# CHECK-NEXT:   8.67317614e-01 -3.72877168e-01 -9.66520965e-01 -6.18431450e-01
# CHECK-NEXT:  -6.35042576e-01 -2.05874304e-02 -2.81237414e-01 -2.90268434e-02
# CHECK-NEXT:  -1.21920058e+00 -5.06691875e-01 -1.13640992e+00  3.45365531e-01
# CHECK-NEXT:  -1.10886242e+00  3.52015143e-01 -6.12350029e-01 -9.87280798e-01
# CHECK-NEXT:   1.09549067e+00  2.38244560e-01 -3.23705638e-01 -1.90151610e+00
# CHECK-NEXT:  -1.78192435e+00 -2.70408185e-01  4.10727263e-01  8.02532175e-02
# CHECK-NEXT:   7.67081636e-01 -1.45651225e+00  5.26951517e-01  9.28822686e-01
# CHECK-NEXT:  -1.55476094e+00 -6.38221832e-01  9.37371770e-02  1.84603368e+00
# CHECK-NEXT:  -1.27088951e+00 -2.16579798e-01  3.26513596e-01  2.08155411e+00
# CHECK-NEXT:  -3.92704859e-01 -1.62973632e+00  2.03739240e+00  6.22174535e-01
# CHECK-NEXT:  -9.51409675e-01 -2.27385803e-01  5.57302413e-01 -3.48671822e-01
# CHECK-NEXT:   1.23447158e-01  8.50850225e-02 -1.32634249e-01  3.13340118e-01
# CHECK-NEXT:   1.10801199e+00  8.22676613e-01  1.34091520e+00  4.71424971e-02
# CHECK-NEXT:  -1.58203615e+00 -1.89406233e+00 -7.01872819e-01  1.36576769e+00
# CHECK-NEXT:  -6.85053208e-02  1.64627120e+00  9.03355076e-01 -3.42040813e-01
# CHECK-NEXT:  -8.67372136e-01  4.80399340e-01 -1.59309443e-01 -3.70875119e-01
# CHECK-NEXT:  -1.08611674e-01  9.80922470e-01 -5.64137888e-01 -1.97664133e+00
# CHECK-NEXT:  -1.76539536e-01  3.70141699e-02 -8.35561926e-01  6.10186220e-01
# CHECK-NEXT:  -1.31177448e+00  3.78866162e-01 -7.24178314e-01  8.70719146e-01
# CHECK-NEXT:   7.84267661e-01 -1.68916986e-01 -1.49940729e+00 -1.11926679e-01
# CHECK-NEXT:  -1.43735635e+00 -3.03024304e-01  1.41841624e+00 -1.14698498e+00
# CHECK-NEXT:   1.25092302e-01  2.05846115e+00  9.19880544e-01  1.45911401e-01
# CHECK-NEXT:   7.76395628e-01  2.12205059e-01  5.12976447e-01 -4.82044154e-01
# CHECK-NEXT:   2.28962851e-01 -1.46967924e+00  4.45720958e-01  1.35042708e+00
# CHECK-NEXT:  -8.11020833e-01 -1.26523134e+00  1.19341208e+00  1.49916039e+00
# CHECK-NEXT:   3.05499398e-01 -1.87835370e-01 -1.20718342e+00  2.22229387e-02
# CHECK-NEXT:   2.09516225e-01  6.16588252e-01  3.66630302e-01  8.44455647e-01
# CHECK-NEXT:   3.00707963e-01  1.07351550e+00  6.26632060e-02  2.42747318e-01
# CHECK-NEXT:   1.92450640e-01  1.28151216e+00  2.13139184e+00  4.79820522e-01
# CHECK-NEXT:  -6.09929456e-01  5.69997315e-01 -8.92470879e-01  5.32505401e-01
# CHECK-NEXT:  -1.27805762e+00  6.64146444e-01 -2.59003939e-01  1.40212845e+00
# CHECK-NEXT:  -5.29828119e-01 -8.21693918e-01 -5.45223483e-01  5.92955962e-03
# CHECK-NEXT:   7.72728613e-01 -8.02005813e-01 -2.58793927e-01  2.30869258e-01
# CHECK-NEXT:   6.99931739e-01 -1.19382207e+00 -1.04729940e+00  1.47284561e-01
# CHECK-NEXT:  -1.09543816e-01  8.72701660e-01 -2.52244485e+00  9.07765293e-01
# CHECK-NEXT:   8.68312904e-02 -3.08580471e-01  9.24489401e-01  6.53436125e-01
# CHECK-NEXT:   1.44498241e+00  1.16908355e+00  8.91607889e-01  1.04583818e+00
# CHECK-NEXT:  -4.03384227e-01 -1.02877504e+00 -1.02129462e-01 -7.83941199e-01
# CHECK-NEXT:  -5.54712692e-01 -1.10646954e+00  1.39718037e+00 -1.80418464e+00
# CHECK-NEXT:  -1.75783869e+00  4.99055716e-01 -9.93633674e-01  7.71198578e-01
# CHECK-NEXT:  -1.55213567e-01  4.97491432e-01  1.04797501e+00  1.65944409e+00
# CHECK-NEXT:   3.45163871e-01  2.42826271e-01 -7.24740467e-01 -8.82135896e-01
# CHECK-NEXT:  -2.15970903e+00 -3.28942722e-01  3.90171458e-01 -2.77940441e-01
# CHECK-NEXT:   8.79364553e-01 -1.23316356e+00  7.13420316e-01 -2.06641354e-01
# CHECK-NEXT:   2.33852818e-01 -1.03188471e+00  9.28470745e-01 -3.86463776e-01
# CHECK-NEXT:   2.27926826e+00  3.29174259e-01  1.60650633e+00 -7.71665475e-01
# CHECK-NEXT:   4.57444266e-01 -6.34015116e-01  1.32202149e+00  1.31007062e+00
# CHECK-NEXT:  -9.32144296e-01 -1.57388916e+00  1.83478348e+00  7.65140445e-01
# CHECK-NEXT:  -9.49527714e-01  2.58097349e-01  8.10902696e-01 -2.61997602e-01
# CHECK-NEXT:  -2.27371244e-01 -3.49378157e-01 -8.14393598e-01 -7.47070143e-01
# CHECK-NEXT:  -1.74348172e-01 -1.01829122e-01 -1.65454777e+00  1.93483760e+00
# CHECK-NEXT:  -4.90001155e-02  1.06824176e+00  2.37726037e-01  2.08337737e+00
# CHECK-NEXT:  -2.56877482e+00 -1.97498342e-01 -4.59035268e-01  6.07068474e-01
# CHECK-NEXT:   3.13199172e-01 -2.20824295e+00  2.37825212e-01  1.67076309e+00
# CHECK-NEXT:   3.54097015e-01  6.70574250e-01  3.35160434e-01  1.02876477e-01
# CHECK-NEXT:  -5.48977809e-01 -1.07213730e-01 -1.42262590e-01 -4.20490640e-01
# CHECK-NEXT:  -7.82277274e-01 -1.77139738e+00 -2.43809350e-03 -1.45442567e+00
# CHECK-NEXT:   1.52608446e+00 -1.73655947e-01  2.44856052e-01 -4.16330047e-01
# CHECK-NEXT:  -1.13590819e+00  2.31811503e-01 -1.27423071e+00  1.32152286e+00
# CHECK-NEXT:  -1.08928646e+00 -8.56091857e-01  1.22903273e-01 -8.77895138e-02
# CHECK-NEXT:  -7.61647686e-01 -6.96667577e-03  5.77938262e-02 -6.50733690e-01
# CHECK-NEXT:  -3.03594111e-01 -1.50179499e+00  1.21830811e+00 -3.56459812e-01
# CHECK-NEXT:  -6.07066136e-02  3.68427669e-03 -1.05091343e+00  9.73330782e-01
# CHECK-NEXT:   7.40786802e-01  1.14890063e+00 -1.09636879e+00  1.23220047e-01
# CHECK-NEXT:  -1.13415721e+00  3.27091445e-01 -9.67616471e-01  2.86635664e-01
# CHECK-NEXT:   1.48698930e+00 -5.85469736e-01  2.61860080e-01  1.43996282e-01
# CHECK-NEXT:   2.16525288e+00  8.36099692e-01 -1.05774265e+00 -2.41912103e-01
# CHECK-NEXT:   2.26028238e-01  3.69156024e-02  7.38699325e-01 -7.32185705e-01
# CHECK-NEXT:  -9.76160342e-01 -1.02653482e+00 -7.09737977e-02  9.09989502e-01
# CHECK-NEXT:   9.81780834e-02  3.44595812e-01  6.34541060e-02  1.04350464e+00
# CHECK-NEXT:  -4.59485802e-02 -4.65768970e-02  1.08762449e+00 -6.53689055e-01
# CHECK-NEXT:  -8.52930286e-01  4.06276719e-01 -1.46785291e+00  1.24663048e+00
# CHECK-NEXT:   6.14678976e-01  5.23488325e-01  3.62678781e-02 -1.24950372e+00
# CHECK-NEXT:  -4.08886362e-01 -8.36645017e-02  1.42921045e+00 -1.18911002e+00
# CHECK-NEXT:   1.70160211e+00 -4.76993975e-01  5.45639826e-01  9.45618619e-02
# CHECK-NEXT:   3.85015245e-01  1.27816791e+00 -7.27717341e-01 -1.30258186e+00
# CHECK-NEXT:   1.53350979e+00 -1.81375618e-01 -8.37233426e-01  1.18251982e+00
# CHECK-NEXT:   2.80790812e-01  7.36278464e-01  8.94895664e-01  9.75620466e-01
# CHECK-NEXT:   2.36919990e+00 -1.21470291e+00 -9.87420670e-01  1.20454984e+00
# CHECK-NEXT:   1.37333692e-01 -2.72810154e-01  6.34002474e-02  1.19467704e+00
# CHECK-NEXT:  -4.47586739e-01  4.11975652e-01 -1.41736030e+00 -1.25011571e+00
# CHECK-NEXT:  -6.05926442e-01 -3.28036275e-01 -5.58047506e-01 -2.58778112e-01
# CHECK-NEXT:   9.79470373e-01 -3.48625875e-01 -1.04851536e-01 -7.99035259e-01
# CHECK-NEXT:  -6.52783989e-01 -1.44069816e+00  3.60692767e-01  4.29090947e-01
# CHECK-NEXT:  -8.96198551e-01  3.87918675e-01 -2.09320526e-01 -3.62665859e-01
# CHECK-NEXT:   7.14834191e-01 -1.41332773e-01 -1.85364420e-01 -1.59320228e-01
# CHECK-NEXT:   3.25294491e-01 -1.37592288e-01  5.13529818e-01 -3.86672187e-01
# CHECK-NEXT:   1.00595513e+00  7.30305463e-01  7.54192676e-01  1.47089852e-01
# CHECK-NEXT:  -6.90359107e-01 -1.21965483e-02  1.15422652e+00 -1.17474077e+00
# CHECK-NEXT:  -7.47355170e-01  1.16031513e+00 -8.17241936e-01 -1.79288296e+00
# CHECK-NEXT:   1.01775230e+00  1.42471274e+00 -2.05749520e-01  4.28659697e-01
# CHECK-NEXT:   7.54530994e-01  6.25818362e-01  9.82836960e-03 -2.46105765e+00
# CHECK-NEXT:   4.51050961e-01 -1.51304924e-01 -3.94770173e-01  9.25950584e-01
# CHECK-NEXT:   7.25375516e-01  1.29639078e+00 -1.91385705e+00 -6.01650466e-01
# CHECK-NEXT:  -5.02117790e-02  1.63046127e-01 -2.95031352e-01 -3.73459693e-01
# CHECK-NEXT:  -1.00243393e+00 -8.80503964e-01  1.06060767e+00  5.87710702e-01
# CHECK-NEXT:  -2.56964002e-01 -1.29705617e+00  6.16055915e-01 -1.00391882e+00
# CHECK-NEXT:  -8.10149810e-01  6.93045071e-01  4.78662161e-01  2.97158545e-01
# CHECK-NEXT:  -2.01822058e+00 -1.93275690e-01 -3.80491598e-02 -2.01453820e+00
# CHECK-NEXT:   1.13793457e+00 -4.05226161e-01 -1.28383065e+00  5.34389446e-01
# CHECK-NEXT:   1.39329553e+00  3.86430224e-02  1.42025550e+00  3.98086202e-01
# CHECK-NEXT:   1.37553985e+00  3.92792635e-01 -4.77273243e-01  1.83391693e-02
# CHECK-NEXT:   2.24710003e-01  9.93993727e-01 -1.73230275e+00 -1.16245902e+00
# CHECK-NEXT:  -7.87493467e-01  1.19610826e+00 -4.06157306e-01 -2.62854362e-02
# CHECK-NEXT:  -1.64395705e+00 -2.16104716e+00 -3.99602753e-01  1.05222156e+00
# CHECK-NEXT:   1.03286068e-01  7.43512228e-01  5.91496657e-01  3.08471108e-01
# CHECK-NEXT:   1.22222857e+00  8.19185430e-02  6.50708598e-01 -9.10108193e-01
# CHECK-NEXT:  -1.04021008e+00  1.07608940e+00 -2.48794748e-01  1.35511969e+00
# CHECK-NEXT:  -9.91019157e-02 -1.62331644e+00  5.48026969e-01 -2.22571770e-01
# CHECK-NEXT:  -3.97097213e-02  9.97434775e-01  1.97734839e+00  1.70949511e-01
# CHECK-NEXT:   2.71601921e+00 -5.71595961e-01  1.23841416e-01  2.02020987e+00
# CHECK-NEXT:  -6.15941493e-01  3.25836464e-01 -1.13270954e+00  7.17599393e-01
# CHECK-NEXT:   1.21842181e+00  1.15349105e+00  9.20713528e-01 -1.85870870e-01
# CHECK-NEXT:   9.49675711e-03  2.07458247e-02 -1.38589741e-01 -6.84951479e-01
# CHECK-NEXT:  -8.25874630e-01 -2.14261253e-01  2.68486040e-01 -8.66494251e-01
# CHECK-NEXT:   4.59265358e-02  9.22162712e-01 -1.41997388e+00  4.07666748e-01
# CHECK-NEXT:   3.21709599e-01  1.14234071e+00 -5.70641877e-01  5.74899505e-02
# CHECK-NEXT:   2.67966147e-02  4.40023510e-01  3.97037120e-03  1.06281152e+00
# CHECK-NEXT:   7.96307670e-01  4.88866037e-01  5.41452375e-01  8.32869103e-01
# CHECK-NEXT:   7.77839850e-01 -2.24269718e-03  3.93943353e-01 -1.08339155e+00
# CHECK-NEXT:   1.31440043e+00 -2.48046331e+00  1.63560857e-01  2.46285869e-01
# CHECK-NEXT:  -2.43436255e+00 -1.19000547e+00 -8.29483545e-01 -2.26680600e+00
# CHECK-NEXT:  -6.06583021e-01  5.07585492e-01 -1.20469550e-01  5.78526153e-01
# CHECK-NEXT:  -2.58197577e-01  3.60226307e-01  9.60450580e-01  1.51439617e+00
# CHECK-NEXT:   1.15863222e+00  2.32527380e-01  1.49282132e+00 -2.65613725e-01
# CHECK-NEXT:   2.47890400e-01 -6.84019191e-01 -6.06476705e-01 -1.27340110e+00
# CHECK-NEXT:  -7.95127089e-02  6.77809604e-01  8.88161842e-02  2.66326445e-01
# CHECK-NEXT:  -6.62399178e-01  1.50255055e+00 -5.93008629e-01  1.30672466e+00
# CHECK-NEXT:  -2.21893927e-01 -6.51561302e-01 -1.57050447e+00  7.69860243e-01
# CHECK-NEXT:   4.11368759e-01  1.87639858e-01 -1.68769198e+00 -9.23028839e-01
# CHECK-NEXT:   9.72028852e-01 -5.56260723e-02 -3.00799177e-01  1.59209218e+00
# CHECK-NEXT:  -1.68361993e-01  1.68002663e+00  2.05607357e-01  6.99531460e-01
# CHECK-NEXT:  -2.82832083e-02 -1.89970569e+00 -9.83200676e-01  4.47907519e-01
# CHECK-NEXT:  -5.70746895e-01 -8.88154888e-01  1.99981995e+00 -2.82082357e-01
# CHECK-NEXT:  -1.71998986e+00  3.22783199e-01  6.09991483e-01 -1.37795661e+00
# CHECK-NEXT:   2.45163524e+00 -6.16594846e-01 -5.25063433e-01 -8.77708288e-01
# CHECK-NEXT:  -7.06681752e-01 -5.40184812e-01  2.26385117e+00 -4.30450921e-01
# CHECK-NEXT:  -9.69422888e-01  3.16025349e-02  3.76275219e-01 -2.50739295e-01
# CHECK-NEXT:  -1.05110104e+00 -5.15198609e-01 -3.58079898e-01  1.08296743e+00
# CHECK-NEXT:  -5.62193248e-01  1.48791132e+00  6.62092812e-01 -5.04033956e-01
# CHECK-NEXT:  -7.94790549e-01 -5.40261542e-01 -1.13677532e+00  5.65917748e-01
# CHECK-NEXT:  -1.51287057e-01 -1.02974152e+00 -5.11068493e-01  1.16380485e+00
# CHECK-NEXT:  -1.23588045e+00 -1.56588701e+00 -3.14901241e-02  1.02838636e+00
# CHECK-NEXT:  -1.13336189e+00 -2.45307758e+00  7.80293872e-01 -1.65232724e+00
# CHECK-NEXT:  -2.90584664e-01 -1.66006795e+00  3.82964791e-01 -1.25463083e+00
# CHECK-NEXT:   7.46177914e-01  2.87266304e+00 -1.57015367e+00 -3.61229384e-01
# CHECK-NEXT:   7.16172571e-01 -1.76834234e-01 -1.10826025e-01 -4.96778122e-01]>: tensor<768xf32>} : () -> tensor<768xf32>
# CHECK-NEXT:   %2740 = "vm.call" @device.matmul(%2771,%2983) : (tensor<16x768xf32>,tensor<768x2304xf32>) -> (tensor<16x2304xf32>)
# CHECK-NEXT:   %2772 = "vm.call" @device.add(%2740,%2984) : (tensor<16x2304xf32>,tensor<2304xf32>) -> (tensor<16x2304xf32>)
# CHECK-NEXT:   %3048 = "vm.call" @device.slice(%2772) : (tensor<16x2304xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3049 = "vm.call" @device.slice(%2772) : (tensor<16x2304xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3050 = "vm.call" @device.slice(%2772) : (tensor<16x2304xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3087 = "vm.call" @device.concat(%3048,%3049,%3050) : (tensor<16x768xf32>,tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<3x16x768xf32>)
# CHECK-NEXT:   %2987 = "vm.rodata" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2895 = "vm.call" @device.gather(%3087,%2987) : (tensor<3x16x768xf32>,tensor<1xi32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3051 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3052 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3053 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3054 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3055 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3056 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3057 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3058 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3059 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3060 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3061 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3062 = "vm.call" @device.slice(%2895) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3088 = "vm.call" @device.concat(%3051,%3052,%3053,%3054,%3055,%3056,%3057,%3058,%3059,%3060,%3061,%3062) : (tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>) -> (tensor<12x16x64xf32>)
# CHECK-NEXT:   %2988 = "vm.rodata" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2896 = "vm.call" @device.gather(%3087,%2988) : (tensor<3x16x768xf32>,tensor<1xi32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3063 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3064 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3065 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3066 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3067 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3068 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3069 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3070 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3071 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3072 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3073 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3074 = "vm.call" @device.slice(%2896) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3089 = "vm.call" @device.concat(%3063,%3064,%3065,%3066,%3067,%3068,%3069,%3070,%3071,%3072,%3073,%3074) : (tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>) -> (tensor<12x16x64xf32>)
# CHECK-NEXT:   %2989 = "vm.rodata" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2897 = "vm.call" @device.gather(%3087,%2989) : (tensor<3x16x768xf32>,tensor<1xi32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3075 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3076 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3077 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3078 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3079 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3080 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3081 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3082 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3083 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3084 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3085 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3086 = "vm.call" @device.slice(%2897) : (tensor<16x768xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3090 = "vm.call" @device.concat(%3075,%3076,%3077,%3078,%3079,%3080,%3081,%3082,%3083,%3084,%3085,%3086) : (tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>) -> (tensor<12x16x64xf32>)
# CHECK-NEXT:   %2587 = "vm.const.i32" () {value = 1: i32} : () -> i32
# CHECK-NEXT:   %2990 = "vm.rodata" () {value = dense<[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0.
# CHECK-NEXT:  0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# CHECK-NEXT:  1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0. 0. 0.
# CHECK-NEXT:  0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# CHECK-NEXT:  1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
# CHECK-NEXT:  0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.
# CHECK-NEXT:  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
# CHECK-NEXT:  1. 1. 1. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0.
# CHECK-NEXT:  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1.
# CHECK-NEXT:  1. 1. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.
# CHECK-NEXT:  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]>: tensor<16x16xi32>} : () -> tensor<16x16xi32>
# CHECK-NEXT:   %2683 = "vm.call" @device.sub(%2587,%2990) : (i32,tensor<16x16xi32>) -> (i32)
# CHECK-NEXT:   %2588 = "vm.const.f32" () {value = -1e-10: f32} : () -> f32
# CHECK-NEXT:   %2684 = "vm.call" @device.mul(%2683,%2588) : (i32,f32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2991 = "vm.rodata" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2898 = "vm.call" @device.gather(%3088,%2991) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2992 = "vm.rodata" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2899 = "vm.call" @device.gather(%3089,%2992) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2993 = "vm.rodata" () {value = dense<[0]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2900 = "vm.call" @device.gather(%3090,%2993) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2589 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2590 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2685 = "vm.call" @device.div(%2589,%2590) : (i32,i32) -> (i32)
# CHECK-NEXT:   %2994 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3105 = "vm.call" @device.transpose(%2899,%2994) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2741 = "vm.call" @device.matmul(%2898,%3105) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2725 = "vm.call" @device.sqrt(%2685) : (i32) -> (i32)
# CHECK-NEXT:   %2701 = "vm.call" @device.splat(%2725) : (i32) -> (i32)
# CHECK-NEXT:   %3122 = "vm.call" @device.div(%2741,%2701) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2773 = "vm.call" @device.add(%3122,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2956 = "vm.call" @device.reduce_max(%2773) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2798 = "vm.call" @device.sub(%2773,%2956) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3092 = "vm.call" @device.exp(%2798) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2938 = "vm.call" @device.reduce_sum(%3092) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3123 = "vm.call" @device.div(%3092,%2938) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2742 = "vm.call" @device.matmul(%3123,%2900) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2995 = "vm.rodata" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2901 = "vm.call" @device.gather(%3088,%2995) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2996 = "vm.rodata" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2902 = "vm.call" @device.gather(%3089,%2996) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2997 = "vm.rodata" () {value = dense<[1]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2903 = "vm.call" @device.gather(%3090,%2997) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2595 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2596 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2686 = "vm.call" @device.div(%2595,%2596) : (i32,i32) -> (i32)
# CHECK-NEXT:   %2998 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3106 = "vm.call" @device.transpose(%2902,%2998) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2743 = "vm.call" @device.matmul(%2901,%3106) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2726 = "vm.call" @device.sqrt(%2686) : (i32) -> (i32)
# CHECK-NEXT:   %2702 = "vm.call" @device.splat(%2726) : (i32) -> (i32)
# CHECK-NEXT:   %3124 = "vm.call" @device.div(%2743,%2702) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2774 = "vm.call" @device.add(%3124,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2957 = "vm.call" @device.reduce_max(%2774) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2799 = "vm.call" @device.sub(%2774,%2957) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3093 = "vm.call" @device.exp(%2799) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2939 = "vm.call" @device.reduce_sum(%3093) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3125 = "vm.call" @device.div(%3093,%2939) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2744 = "vm.call" @device.matmul(%3125,%2903) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2999 = "vm.rodata" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2904 = "vm.call" @device.gather(%3088,%2999) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3000 = "vm.rodata" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2905 = "vm.call" @device.gather(%3089,%3000) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3001 = "vm.rodata" () {value = dense<[2]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2906 = "vm.call" @device.gather(%3090,%3001) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2601 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2602 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2687 = "vm.call" @device.div(%2601,%2602) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3002 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3107 = "vm.call" @device.transpose(%2905,%3002) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2745 = "vm.call" @device.matmul(%2904,%3107) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2727 = "vm.call" @device.sqrt(%2687) : (i32) -> (i32)
# CHECK-NEXT:   %2703 = "vm.call" @device.splat(%2727) : (i32) -> (i32)
# CHECK-NEXT:   %3126 = "vm.call" @device.div(%2745,%2703) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2775 = "vm.call" @device.add(%3126,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2958 = "vm.call" @device.reduce_max(%2775) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2800 = "vm.call" @device.sub(%2775,%2958) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3094 = "vm.call" @device.exp(%2800) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2940 = "vm.call" @device.reduce_sum(%3094) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3127 = "vm.call" @device.div(%3094,%2940) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2746 = "vm.call" @device.matmul(%3127,%2906) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3003 = "vm.rodata" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2907 = "vm.call" @device.gather(%3088,%3003) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3004 = "vm.rodata" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2908 = "vm.call" @device.gather(%3089,%3004) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3005 = "vm.rodata" () {value = dense<[3]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2909 = "vm.call" @device.gather(%3090,%3005) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2607 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2608 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2688 = "vm.call" @device.div(%2607,%2608) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3006 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3108 = "vm.call" @device.transpose(%2908,%3006) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2747 = "vm.call" @device.matmul(%2907,%3108) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2728 = "vm.call" @device.sqrt(%2688) : (i32) -> (i32)
# CHECK-NEXT:   %2704 = "vm.call" @device.splat(%2728) : (i32) -> (i32)
# CHECK-NEXT:   %3128 = "vm.call" @device.div(%2747,%2704) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2776 = "vm.call" @device.add(%3128,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2959 = "vm.call" @device.reduce_max(%2776) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2801 = "vm.call" @device.sub(%2776,%2959) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3095 = "vm.call" @device.exp(%2801) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2941 = "vm.call" @device.reduce_sum(%3095) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3129 = "vm.call" @device.div(%3095,%2941) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2748 = "vm.call" @device.matmul(%3129,%2909) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3007 = "vm.rodata" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2910 = "vm.call" @device.gather(%3088,%3007) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3008 = "vm.rodata" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2911 = "vm.call" @device.gather(%3089,%3008) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3009 = "vm.rodata" () {value = dense<[4]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2912 = "vm.call" @device.gather(%3090,%3009) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2613 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2614 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2689 = "vm.call" @device.div(%2613,%2614) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3010 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3109 = "vm.call" @device.transpose(%2911,%3010) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2749 = "vm.call" @device.matmul(%2910,%3109) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2729 = "vm.call" @device.sqrt(%2689) : (i32) -> (i32)
# CHECK-NEXT:   %2705 = "vm.call" @device.splat(%2729) : (i32) -> (i32)
# CHECK-NEXT:   %3130 = "vm.call" @device.div(%2749,%2705) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2777 = "vm.call" @device.add(%3130,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2960 = "vm.call" @device.reduce_max(%2777) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2802 = "vm.call" @device.sub(%2777,%2960) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3096 = "vm.call" @device.exp(%2802) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2942 = "vm.call" @device.reduce_sum(%3096) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3131 = "vm.call" @device.div(%3096,%2942) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2750 = "vm.call" @device.matmul(%3131,%2912) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3011 = "vm.rodata" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2913 = "vm.call" @device.gather(%3088,%3011) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3012 = "vm.rodata" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2914 = "vm.call" @device.gather(%3089,%3012) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3013 = "vm.rodata" () {value = dense<[5]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2915 = "vm.call" @device.gather(%3090,%3013) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2619 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2620 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2690 = "vm.call" @device.div(%2619,%2620) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3014 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3110 = "vm.call" @device.transpose(%2914,%3014) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2751 = "vm.call" @device.matmul(%2913,%3110) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2730 = "vm.call" @device.sqrt(%2690) : (i32) -> (i32)
# CHECK-NEXT:   %2706 = "vm.call" @device.splat(%2730) : (i32) -> (i32)
# CHECK-NEXT:   %3132 = "vm.call" @device.div(%2751,%2706) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2778 = "vm.call" @device.add(%3132,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2961 = "vm.call" @device.reduce_max(%2778) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2803 = "vm.call" @device.sub(%2778,%2961) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3097 = "vm.call" @device.exp(%2803) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2943 = "vm.call" @device.reduce_sum(%3097) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3133 = "vm.call" @device.div(%3097,%2943) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2752 = "vm.call" @device.matmul(%3133,%2915) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3015 = "vm.rodata" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2916 = "vm.call" @device.gather(%3088,%3015) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3016 = "vm.rodata" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2917 = "vm.call" @device.gather(%3089,%3016) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3017 = "vm.rodata" () {value = dense<[6]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2918 = "vm.call" @device.gather(%3090,%3017) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2625 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2626 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2691 = "vm.call" @device.div(%2625,%2626) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3018 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3111 = "vm.call" @device.transpose(%2917,%3018) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2753 = "vm.call" @device.matmul(%2916,%3111) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2731 = "vm.call" @device.sqrt(%2691) : (i32) -> (i32)
# CHECK-NEXT:   %2707 = "vm.call" @device.splat(%2731) : (i32) -> (i32)
# CHECK-NEXT:   %3134 = "vm.call" @device.div(%2753,%2707) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2779 = "vm.call" @device.add(%3134,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2962 = "vm.call" @device.reduce_max(%2779) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2804 = "vm.call" @device.sub(%2779,%2962) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3098 = "vm.call" @device.exp(%2804) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2944 = "vm.call" @device.reduce_sum(%3098) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3135 = "vm.call" @device.div(%3098,%2944) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2754 = "vm.call" @device.matmul(%3135,%2918) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3019 = "vm.rodata" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2919 = "vm.call" @device.gather(%3088,%3019) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3020 = "vm.rodata" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2920 = "vm.call" @device.gather(%3089,%3020) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3021 = "vm.rodata" () {value = dense<[7]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2921 = "vm.call" @device.gather(%3090,%3021) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2631 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2632 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2692 = "vm.call" @device.div(%2631,%2632) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3022 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3112 = "vm.call" @device.transpose(%2920,%3022) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2755 = "vm.call" @device.matmul(%2919,%3112) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2732 = "vm.call" @device.sqrt(%2692) : (i32) -> (i32)
# CHECK-NEXT:   %2708 = "vm.call" @device.splat(%2732) : (i32) -> (i32)
# CHECK-NEXT:   %3136 = "vm.call" @device.div(%2755,%2708) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2780 = "vm.call" @device.add(%3136,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2963 = "vm.call" @device.reduce_max(%2780) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2805 = "vm.call" @device.sub(%2780,%2963) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3099 = "vm.call" @device.exp(%2805) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2945 = "vm.call" @device.reduce_sum(%3099) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3137 = "vm.call" @device.div(%3099,%2945) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2756 = "vm.call" @device.matmul(%3137,%2921) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3023 = "vm.rodata" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2922 = "vm.call" @device.gather(%3088,%3023) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3024 = "vm.rodata" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2923 = "vm.call" @device.gather(%3089,%3024) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3025 = "vm.rodata" () {value = dense<[8]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2924 = "vm.call" @device.gather(%3090,%3025) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2637 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2638 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2693 = "vm.call" @device.div(%2637,%2638) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3026 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3113 = "vm.call" @device.transpose(%2923,%3026) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2757 = "vm.call" @device.matmul(%2922,%3113) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2733 = "vm.call" @device.sqrt(%2693) : (i32) -> (i32)
# CHECK-NEXT:   %2709 = "vm.call" @device.splat(%2733) : (i32) -> (i32)
# CHECK-NEXT:   %3138 = "vm.call" @device.div(%2757,%2709) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2781 = "vm.call" @device.add(%3138,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2964 = "vm.call" @device.reduce_max(%2781) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2806 = "vm.call" @device.sub(%2781,%2964) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3100 = "vm.call" @device.exp(%2806) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2946 = "vm.call" @device.reduce_sum(%3100) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3139 = "vm.call" @device.div(%3100,%2946) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2758 = "vm.call" @device.matmul(%3139,%2924) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3027 = "vm.rodata" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2925 = "vm.call" @device.gather(%3088,%3027) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3028 = "vm.rodata" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2926 = "vm.call" @device.gather(%3089,%3028) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3029 = "vm.rodata" () {value = dense<[9]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2927 = "vm.call" @device.gather(%3090,%3029) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2643 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2644 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2694 = "vm.call" @device.div(%2643,%2644) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3030 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3114 = "vm.call" @device.transpose(%2926,%3030) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2759 = "vm.call" @device.matmul(%2925,%3114) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2734 = "vm.call" @device.sqrt(%2694) : (i32) -> (i32)
# CHECK-NEXT:   %2710 = "vm.call" @device.splat(%2734) : (i32) -> (i32)
# CHECK-NEXT:   %3140 = "vm.call" @device.div(%2759,%2710) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2782 = "vm.call" @device.add(%3140,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2965 = "vm.call" @device.reduce_max(%2782) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2807 = "vm.call" @device.sub(%2782,%2965) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3101 = "vm.call" @device.exp(%2807) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2947 = "vm.call" @device.reduce_sum(%3101) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3141 = "vm.call" @device.div(%3101,%2947) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2760 = "vm.call" @device.matmul(%3141,%2927) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3031 = "vm.rodata" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2928 = "vm.call" @device.gather(%3088,%3031) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3032 = "vm.rodata" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2929 = "vm.call" @device.gather(%3089,%3032) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3033 = "vm.rodata" () {value = dense<[10]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2930 = "vm.call" @device.gather(%3090,%3033) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2649 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2650 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2695 = "vm.call" @device.div(%2649,%2650) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3034 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3115 = "vm.call" @device.transpose(%2929,%3034) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2761 = "vm.call" @device.matmul(%2928,%3115) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2735 = "vm.call" @device.sqrt(%2695) : (i32) -> (i32)
# CHECK-NEXT:   %2711 = "vm.call" @device.splat(%2735) : (i32) -> (i32)
# CHECK-NEXT:   %3142 = "vm.call" @device.div(%2761,%2711) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2783 = "vm.call" @device.add(%3142,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2966 = "vm.call" @device.reduce_max(%2783) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2808 = "vm.call" @device.sub(%2783,%2966) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3102 = "vm.call" @device.exp(%2808) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2948 = "vm.call" @device.reduce_sum(%3102) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3143 = "vm.call" @device.div(%3102,%2948) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2762 = "vm.call" @device.matmul(%3143,%2930) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3035 = "vm.rodata" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2931 = "vm.call" @device.gather(%3088,%3035) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3036 = "vm.rodata" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2932 = "vm.call" @device.gather(%3089,%3036) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3037 = "vm.rodata" () {value = dense<[11]>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2933 = "vm.call" @device.gather(%3090,%3037) : (tensor<12x16x64xf32>,tensor<1xi32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %2655 = "vm.const.i32" () {value = 768: i32} : () -> i32
# CHECK-NEXT:   %2656 = "vm.const.i32" () {value = 12: i32} : () -> i32
# CHECK-NEXT:   %2696 = "vm.call" @device.div(%2655,%2656) : (i32,i32) -> (i32)
# CHECK-NEXT:   %3038 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3116 = "vm.call" @device.transpose(%2932,%3038) : (tensor<16x64xf32>,tensor<2xi32>) -> (tensor<64x16xf32>)
# CHECK-NEXT:   %2763 = "vm.call" @device.matmul(%2931,%3116) : (tensor<16x64xf32>,tensor<64x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2736 = "vm.call" @device.sqrt(%2696) : (i32) -> (i32)
# CHECK-NEXT:   %2712 = "vm.call" @device.splat(%2736) : (i32) -> (i32)
# CHECK-NEXT:   %3144 = "vm.call" @device.div(%2763,%2712) : (tensor<16x16xf32>,i32) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2784 = "vm.call" @device.add(%3144,%2684) : (tensor<16x16xf32>,tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2967 = "vm.call" @device.reduce_max(%2784) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2809 = "vm.call" @device.sub(%2784,%2967) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %3103 = "vm.call" @device.exp(%2809) : (tensor<16x16xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2949 = "vm.call" @device.reduce_sum(%3103) : (tensor<16x16xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3145 = "vm.call" @device.div(%3103,%2949) : (tensor<16x16xf32>,tensor<16x1xf32>) -> (tensor<16x16xf32>)
# CHECK-NEXT:   %2764 = "vm.call" @device.matmul(%3145,%2933) : (tensor<16x16xf32>,tensor<16x64xf32>) -> (tensor<16x64xf32>)
# CHECK-NEXT:   %3091 = "vm.call" @device.concat(%2742,%2744,%2746,%2748,%2750,%2752,%2754,%2756,%2758,%2760,%2762,%2764) : (tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>,tensor<16x64xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2765 = "vm.call" @device.matmul(%3091,%2985) : (tensor<16x768xf32>,tensor<768x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2785 = "vm.call" @device.add(%2765,%2986) : (tensor<16x768xf32>,tensor<768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2786 = "vm.call" @device.add(%2769,%2785) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2661 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2662 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2663 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2950 = "vm.call" @device.reduce_sum(%2786) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3039 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2971 = "vm.call" @device.cast(%3039) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3146 = "vm.call" @device.div(%2950,%2971) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2951 = "vm.call" @device.reduce_sum(%2786) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3040 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2972 = "vm.call" @device.cast(%3040) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3148 = "vm.call" @device.div(%2951,%2972) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2973 = "vm.call" @device.cast(%3148) : (tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2810 = "vm.call" @device.sub(%2786,%2973) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2846 = "vm.call" @device.mul(%2810,%2810) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2952 = "vm.call" @device.reduce_sum(%2786) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3147 = "vm.call" @device.div(%2952,%2972) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2811 = "vm.call" @device.sub(%2786,%3146) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2713 = "vm.call" @device.splat(%2662) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2848 = "vm.call" @device.mul(%2713,%2811) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2714 = "vm.call" @device.splat(%2661) : (f32) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2787 = "vm.call" @device.add(%3147,%2714) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2737 = "vm.call" @device.sqrt(%2787) : (tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3149 = "vm.call" @device.div(%2848,%2737) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2715 = "vm.call" @device.splat(%2663) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2788 = "vm.call" @device.add(%3149,%2715) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3041 = "vm.rodata" () {value = dense<[ 0.83138123  0.13229829 -0.26965598 ... -0.65889418  0.60237776
# CHECK-NEXT:   0.00700748]>: tensor<768x3072xf32>} : () -> tensor<768x3072xf32>
# CHECK-NEXT:   %3042 = "vm.rodata" () {value = dense<[-0.42023735 -0.1593114   1.08283664 ... -0.1974762   0.7747443
# CHECK-NEXT:   0.03738274]>: tensor<3072xf32>} : () -> tensor<3072xf32>
# CHECK-NEXT:   %3043 = "vm.rodata" () {value = dense<[ 0.86099537  1.1382183  -0.43527711 ...  1.17450839 -0.04072113
# CHECK-NEXT:  -1.14175187]>: tensor<3072x768xf32>} : () -> tensor<3072x768xf32>
# CHECK-NEXT:   %3044 = "vm.rodata" () {value = dense<[-9.64082969e-01  7.24688409e-01  1.20424772e+00 -1.75425184e+00
# CHECK-NEXT:   5.82823959e-02  1.05321161e+00  5.54405845e-01  9.15340339e-01
# CHECK-NEXT:   2.00496542e+00 -1.65930357e+00 -6.99079962e-01 -8.82322676e-02
# CHECK-NEXT:  -8.88625356e-01 -3.29347667e-01  1.23499164e+00  2.79654104e-01
# CHECK-NEXT:  -1.03082813e+00 -1.08548459e+00 -4.80779360e-02  4.04081215e-01
# CHECK-NEXT:  -1.01429728e+00  1.13928716e+00  2.45070675e-01  1.10388842e+00
# CHECK-NEXT:  -2.36209917e-01  1.98557414e-01 -8.17477661e-01  1.66764653e+00
# CHECK-NEXT:  -2.16039945e+00 -6.26161327e-01 -1.12527387e+00 -1.90794782e+00
# CHECK-NEXT:   8.27599269e-01  1.39314520e+00  1.86761186e-01  2.66908659e+00
# CHECK-NEXT:  -2.99505613e-01  1.58010946e-01  1.52922155e+00 -8.98413716e-01
# CHECK-NEXT:  -3.71189908e-03  1.35780793e-01  1.13295111e+00 -1.75693973e+00
# CHECK-NEXT:   4.84263743e-01 -1.95021989e+00  6.79402769e-01  2.57455443e-01
# CHECK-NEXT:  -7.51658521e-01  8.27424810e-01  1.34685083e+00  1.09498518e+00
# CHECK-NEXT:  -1.74184819e+00  2.16119378e-01 -8.26987178e-01 -6.29224200e-02
# CHECK-NEXT:   6.71365831e-01 -4.24111668e-04 -1.71465709e+00 -1.45870189e-01
# CHECK-NEXT:  -3.99571792e-01  2.60475536e-01 -6.06759170e-01  4.59173268e-02
# CHECK-NEXT:   1.58343980e+00  1.25679301e+00  4.36184920e-01 -1.69926262e+00
# CHECK-NEXT:  -1.32595508e+00  2.32646365e-01 -1.19453012e+00  1.93436985e-01
# CHECK-NEXT:  -2.93864123e-01 -1.15448499e+00  4.81423752e-01  5.90412923e-01
# CHECK-NEXT:   2.30310325e-01  1.29766740e+00 -6.74635334e-01 -8.63251583e-01
# CHECK-NEXT:  -1.42651917e+00  1.30015188e+00  1.55434031e-01 -9.75937866e-01
# CHECK-NEXT:  -1.14324353e+00 -1.10511788e+00  7.04762995e-01 -6.21955868e-02
# CHECK-NEXT:   7.14462430e-01 -8.85275184e-01  6.00984427e-01  3.88645856e-01
# CHECK-NEXT:   1.05498156e+00  4.99333251e-02  1.06524915e+00  4.53207353e-01
# CHECK-NEXT:   7.95409093e-01  4.65384658e-01  2.39472057e+00 -2.40496545e-01
# CHECK-NEXT:   1.14909499e-01 -2.80270550e-01 -2.29499528e-01 -1.13811209e+00
# CHECK-NEXT:   2.56160423e-01 -5.25329221e-01 -9.42701122e-02 -4.66762461e-01
# CHECK-NEXT:  -5.85192093e-01 -1.98822551e+00  1.72980523e+00 -1.64215054e+00
# CHECK-NEXT:   1.94325413e+00 -4.22302607e-01  4.69117360e-01  1.26246890e+00
# CHECK-NEXT:   7.12413222e-01 -1.11488230e+00 -7.78497351e-01 -3.41995197e-01
# CHECK-NEXT:  -7.20787613e-01  4.75269847e-01 -1.04181318e+00  1.29647034e+00
# CHECK-NEXT:  -4.90620685e-01  2.12604612e+00 -6.86798268e-01 -8.64538150e-01
# CHECK-NEXT:   1.16485679e+00  2.20888766e-01 -3.93480508e-01 -7.39315946e-01
# CHECK-NEXT:   5.17459061e-01 -8.87976653e-01  9.20829071e-01 -1.51426174e-01
# CHECK-NEXT:   9.79975160e-01  2.25804134e-01 -6.40377908e-01 -5.45306441e-01
# CHECK-NEXT:  -2.34674585e+00  4.98078749e-01 -1.24226950e-01 -1.17674140e+00
# CHECK-NEXT:   1.77692449e+00 -3.07759257e-01  6.01705103e-01 -9.89832600e-01
# CHECK-NEXT:   2.53182320e-01  1.80879235e-01 -7.31368583e-01 -1.46310312e+00
# CHECK-NEXT:  -1.26123708e+00 -2.06348932e-01 -9.13440356e-01 -4.78813962e-01
# CHECK-NEXT:  -6.27269476e-03 -4.83697140e-01 -2.56950058e-01 -5.74839353e-01
# CHECK-NEXT:   2.35758875e+00  1.15677259e-01 -3.28146454e+00  1.30069300e-01
# CHECK-NEXT:  -1.47782642e+00 -3.15299817e-01  8.78555275e-01  1.31704131e-01
# CHECK-NEXT:  -1.10033141e+00  1.23870818e-01 -2.48664133e+00 -1.27434318e+00
# CHECK-NEXT:   3.50780547e-01 -1.00605438e+00 -1.47199982e+00  4.57524607e-01
# CHECK-NEXT:   1.28029200e+00  1.15444764e+00 -1.08698406e+00  3.40378706e-01
# CHECK-NEXT:  -5.94098525e-01 -6.79351266e-01 -3.78736394e-01 -3.17185017e-01
# CHECK-NEXT:  -1.55427147e+00  7.46366114e-02  5.43526408e-02 -2.37072118e+00
# CHECK-NEXT:   2.38596528e+00  2.21173451e+00  4.88104312e-02  1.27097566e+00
# CHECK-NEXT:   1.94453304e-01 -5.18312021e-02 -2.68737038e+00  2.08610902e+00
# CHECK-NEXT:  -4.28808680e-02  2.17957591e+00 -1.44071466e+00  6.71179206e-01
# CHECK-NEXT:   1.33528669e+00  2.41880112e-01 -7.19313691e-01  1.50598671e+00
# CHECK-NEXT:  -9.05982902e-01  9.75053510e-01 -1.38975553e+00  5.58511430e-01
# CHECK-NEXT:  -9.12738845e-01  1.49464467e+00 -3.29358650e-01 -2.43027472e+00
# CHECK-NEXT:  -3.13397177e-01 -4.49816903e-01  2.66576558e-01  9.15658857e-01
# CHECK-NEXT:  -3.45285161e-01 -4.65981839e-01  5.07085023e-01  2.05124228e+00
# CHECK-NEXT:  -7.77841419e-01 -1.84815749e+00  1.94825192e-01 -6.02893024e-01
# CHECK-NEXT:  -4.55518490e-01 -1.74290149e+00 -7.53226967e-01 -5.46781585e-01
# CHECK-NEXT:  -8.85402731e-01 -1.09302635e+00 -3.14830618e-01  1.54663348e-01
# CHECK-NEXT:  -7.30445169e-01 -1.48528826e+00  1.63003682e+00  1.72971295e+00
# CHECK-NEXT:   6.89266924e-01 -1.98789038e+00 -6.30248931e-01  3.13965879e-01
# CHECK-NEXT:  -7.41702564e-01  2.42791102e-01  5.06235181e-01 -8.57496619e-01
# CHECK-NEXT:  -5.86406152e-01 -7.05063684e-01 -4.71059283e-01 -6.59680349e-01
# CHECK-NEXT:   1.15724865e+00  8.39949361e-01 -1.00378521e+00 -5.99746551e-01
# CHECK-NEXT:   3.92284093e-01 -4.53328930e-01  7.83383774e-01  2.17060836e-01
# CHECK-NEXT:   1.12804594e+00  3.12195010e-01  1.54199543e+00  2.49069438e+00
# CHECK-NEXT:   6.95500904e-01  8.77466149e-02 -9.88379217e-01  8.08074660e-01
# CHECK-NEXT:  -1.52690761e-01  1.21000673e+00 -1.12203220e+00 -1.19378582e+00
# CHECK-NEXT:   4.55897526e-01  5.65043102e-03 -1.37307604e+00 -4.14162455e-01
# CHECK-NEXT:  -1.12258119e+00 -1.44002319e-01 -2.06250304e+00 -3.57788119e-01
# CHECK-NEXT:  -1.04511691e+00 -5.34682828e-01 -1.78613812e+00 -3.90543619e-02
# CHECK-NEXT:   3.12691505e-01  6.79034528e-02  3.55992321e-01 -7.42185187e-01
# CHECK-NEXT:  -3.75941752e-02  6.06559225e-01 -9.80593940e-01  3.12755710e-01
# CHECK-NEXT:   4.89153461e-01  5.73569796e-02 -1.51967627e+00 -9.79182358e-01
# CHECK-NEXT:  -2.91585601e-01  2.79575046e-01 -2.40211055e-01 -1.40826789e+00
# CHECK-NEXT:   2.69045584e-01  1.50436117e+00 -1.50837096e+00  9.20578634e-01
# CHECK-NEXT:  -6.70729370e-01 -6.70981065e-01  2.39902438e+00  8.56254657e-01
# CHECK-NEXT:  -1.02224335e+00 -3.58289381e-01 -8.75934441e-01 -3.25264593e-01
# CHECK-NEXT:   3.58972631e-01 -3.86667464e-01  1.87816876e+00 -3.61985197e-01
# CHECK-NEXT:  -5.76951049e-01  1.83793529e+00 -5.80185200e-01 -5.16555580e-01
# CHECK-NEXT:  -6.27443547e-01  4.62819288e-01  1.39826596e+00  3.74017183e-01
# CHECK-NEXT:   8.13238900e-01 -8.29628844e-01 -1.13414240e+00 -8.71945312e-02
# CHECK-NEXT:   3.42506216e-01 -8.74286203e-01 -4.91169490e-01  6.56797971e-01
# CHECK-NEXT:   7.78475859e-01  2.56833353e-01 -6.73869874e-02 -1.40877492e-01
# CHECK-NEXT:   6.25580825e-01 -1.36960992e+00 -5.40039438e-01 -3.18186242e-01
# CHECK-NEXT:  -2.66248764e-01  5.16878249e-01  4.82000975e-01 -5.22597038e-02
# CHECK-NEXT:  -5.59702007e-01  1.18929889e+00 -5.31144724e-02 -7.59281710e-01
# CHECK-NEXT:  -2.75135463e-01  7.15635987e-01  4.46250421e-02 -1.97815644e-01
# CHECK-NEXT:   2.01828410e+00 -5.58505240e-02 -1.69214558e+00  6.44376070e-01
# CHECK-NEXT:  -2.74431884e-01 -2.03257729e+00 -6.48621097e-01  5.98447969e-02
# CHECK-NEXT:  -6.29534611e-01 -1.91449717e-01  6.61770528e-01 -1.77628586e-01
# CHECK-NEXT:   6.61911692e-01  3.94852676e-01 -1.92498988e-01 -2.31887004e-01
# CHECK-NEXT:   4.23575272e-01  7.73879330e-02 -2.06572896e+00  5.26227149e-01
# CHECK-NEXT:   1.74612391e-01 -3.21336923e-01 -1.24333888e+00 -2.01897683e-02
# CHECK-NEXT:   8.37488903e-01  4.59821683e-01  1.27678587e+00 -3.00884912e-02
# CHECK-NEXT:  -1.99792074e-01 -4.41248358e-01  3.08382220e-01  7.12809211e-01
# CHECK-NEXT:  -6.79011765e-01  3.15571676e-01  2.72703119e-01 -1.75371780e+00
# CHECK-NEXT:  -1.54539756e+00  1.05257332e+00  8.49621890e-01  4.75518909e-01
# CHECK-NEXT:  -6.99032637e-01  2.41253775e-01 -4.87432352e-01  1.19039406e+00
# CHECK-NEXT:  -2.40447660e+00  1.08582216e+00 -9.71817396e-02 -2.31214845e+00
# CHECK-NEXT:  -2.51021531e+00 -1.73898844e+00 -3.30242395e-01 -1.38351718e-01
# CHECK-NEXT:   6.58423453e-01 -7.35647530e-01 -1.01189002e-01  5.67099101e-01
# CHECK-NEXT:   3.18072928e-01  1.52540750e+00  1.47480771e-01  1.22746874e+00
# CHECK-NEXT:   1.13792855e+00  5.76477074e-02 -1.86381680e-01 -1.20831449e+00
# CHECK-NEXT:   1.04826636e+00  1.62592239e+00  2.19700388e+00  1.10634225e+00
# CHECK-NEXT:   1.22943012e-01  5.51709103e-01 -1.66506161e+00  8.73067323e-01
# CHECK-NEXT:   2.33408189e-01 -1.32418514e-01 -1.24990609e+00  9.35575396e-01
# CHECK-NEXT:  -1.27255657e-01 -1.92410799e+00  8.84897786e-01 -1.32395772e-01
# CHECK-NEXT:  -6.03391127e-01  1.53795649e-01  2.10961341e+00 -9.48575080e-02
# CHECK-NEXT:   1.34242199e+00 -3.32272783e-01 -4.28759348e-02  1.15159935e-01
# CHECK-NEXT:  -6.67804259e-01 -3.53934794e-01 -1.24730681e+00  7.86364939e-01
# CHECK-NEXT:  -1.05665243e+00  1.94937409e+00 -6.32215703e-02 -1.29522023e+00
# CHECK-NEXT:  -2.76625562e-01  5.64131903e-02  4.39116552e-01 -1.10957546e+00
# CHECK-NEXT:   5.95468470e-01 -7.93513880e-01 -9.07834789e-01  2.97739780e-02
# CHECK-NEXT:  -2.14036172e-01 -7.91807841e-01 -1.86626818e+00 -4.19343736e-01
# CHECK-NEXT:  -1.33254027e+00  8.95769108e-02 -2.19802112e+00  4.38661891e-01
# CHECK-NEXT:  -3.34990155e-01  3.92451897e-01  2.01199183e+00  1.82854049e-01
# CHECK-NEXT:  -2.36005176e-01  1.68063852e+00 -1.15117962e+00  1.89617541e+00
# CHECK-NEXT:  -7.11483210e-01  6.90982453e-01 -1.55533579e-02  3.40210207e-01
# CHECK-NEXT:   1.95939758e+00  1.73698002e+00 -6.81171618e-01  9.34952738e-01
# CHECK-NEXT:  -7.97789466e-01 -1.58390782e+00  1.17497864e+00 -1.46006788e+00
# CHECK-NEXT:  -1.31196318e-01  2.13958503e-01 -5.16896727e-01 -6.91374535e-02
# CHECK-NEXT:  -3.00134760e-01  1.03610304e+00  1.64762650e+00 -3.93395822e-01
# CHECK-NEXT:  -1.93401364e-01 -9.35080235e-01 -1.06745897e+00 -4.32358179e-01
# CHECK-NEXT:   9.34107891e-02  5.98472247e-02 -5.85387013e-01 -7.45541221e-01
# CHECK-NEXT:   3.46752050e-01  1.85462371e-01 -7.44651565e-01  9.70484728e-02
# CHECK-NEXT:   4.59361915e-01  2.50673072e-01 -7.00178520e-01 -1.18401435e-02
# CHECK-NEXT:  -9.10834760e-01  8.27886949e-01 -1.83947092e+00 -1.04070514e+00
# CHECK-NEXT:  -1.08949075e+00 -2.43324838e+00  9.19551440e-01  7.66501717e-01
# CHECK-NEXT:  -4.71956247e-01  1.05233779e+00  1.08205940e+00 -1.88532051e+00
# CHECK-NEXT:  -1.19858670e+00 -1.23330890e+00 -1.32458397e+00  1.43833993e+00
# CHECK-NEXT:  -1.98747576e+00 -9.18466783e-01 -1.57841385e+00 -1.14391338e-01
# CHECK-NEXT:  -1.65493484e-01 -7.74793389e-01 -1.78900414e+00 -6.85663344e-02
# CHECK-NEXT:   1.18017195e+00  5.58556167e-01  1.74954449e-02 -2.94372363e-01
# CHECK-NEXT:  -5.92314560e-01 -6.47152028e-01 -2.06744019e+00  5.74528529e-01
# CHECK-NEXT:   3.93991091e-02 -9.20240273e-01  9.03355186e-02  7.88089937e-01
# CHECK-NEXT:  -1.88481007e-01 -2.48959784e-01 -1.46712795e+00  6.42526830e-01
# CHECK-NEXT:  -3.63834434e-01 -3.43850490e-01 -9.46400284e-02  1.13608794e+00
# CHECK-NEXT:   3.68413947e-01  3.64806474e-01 -3.24905938e-01  1.74183745e+00
# CHECK-NEXT:  -4.66521211e-01  1.59219443e-02  1.28798972e+00  3.23577740e-01
# CHECK-NEXT:   2.09849694e-01 -1.18269503e+00 -1.11643714e+00 -7.93709067e-01
# CHECK-NEXT:  -1.81509687e+00 -6.73485449e-01 -1.96790461e+00  1.10257594e+00
# CHECK-NEXT:  -1.01848533e+00 -2.59545918e+00  1.44805321e+00  7.05314458e-01
# CHECK-NEXT:  -7.46065745e-01 -2.30731951e+00 -7.88145095e-02 -5.01270494e-01
# CHECK-NEXT:   4.53019259e-01 -4.42022105e-01  1.50581317e+00 -9.95201233e-01
# CHECK-NEXT:  -1.31197493e-01 -1.07352408e+00 -2.99551304e-02  8.06361800e-01
# CHECK-NEXT:   1.53943169e+00  7.08223924e-03 -1.87184945e-01 -1.21154073e+00
# CHECK-NEXT:   1.46403134e+00  5.91000848e-01  7.06863528e-01  1.88034222e+00
# CHECK-NEXT:  -1.40999863e+00  1.67231034e-01  7.47405204e-01  3.60032211e-01
# CHECK-NEXT:  -4.28065898e-01 -6.05001645e-01 -7.44388803e-01  1.84452395e+00
# CHECK-NEXT:  -8.89948322e-01 -5.67513968e-01 -1.98115243e-01  1.06847489e+00
# CHECK-NEXT:   1.04840906e+00 -1.17134725e+00  1.05675415e+00  8.85676782e-01
# CHECK-NEXT:  -1.19190560e-01 -1.98193694e-01 -1.20572248e+00  1.24115669e+00
# CHECK-NEXT:  -5.02856100e-01  1.09016621e-01  5.25262012e-01  2.91341057e-01
# CHECK-NEXT:  -7.55499404e-01  9.32517786e-01  2.04031468e-01 -5.04599615e-01
# CHECK-NEXT:  -2.04792826e+00 -2.18455240e+00  5.63839754e-02  1.35731531e+00
# CHECK-NEXT:  -1.24090947e+00  1.36277154e+00 -7.04471910e-01 -1.64765477e+00
# CHECK-NEXT:  -5.85209192e-01  2.15159389e-02  7.77248780e-01 -6.58023744e-01
# CHECK-NEXT:  -4.27492972e-01  1.89171908e-01 -1.49040506e+00 -5.56213921e-02
# CHECK-NEXT:  -5.75272551e-01 -4.68337739e-01 -4.10498616e-01 -9.49777332e-01
# CHECK-NEXT:  -9.68195317e-01  7.23983644e-01  7.29149668e-01  5.21384618e-01
# CHECK-NEXT:  -2.06862639e+00 -1.40695034e+00  7.40687614e-01  3.39983288e-01
# CHECK-NEXT:  -1.98220421e+00  1.30796915e+00 -2.24953377e+00 -1.06983309e+00
# CHECK-NEXT:  -5.91017796e-01  1.35977704e+00 -4.81560176e-01 -7.67588928e-01
# CHECK-NEXT:   1.99526141e-01 -9.42958522e-01  1.58099994e+00 -7.81561076e-02
# CHECK-NEXT:  -4.16294744e-01  5.68511749e-01  6.54341557e-01 -1.60282153e+00
# CHECK-NEXT:   9.62538890e-01  1.82693573e-01  4.92132721e-01  3.72858185e-01
# CHECK-NEXT:  -8.73034464e-01 -3.81891287e-01  3.76457532e-01  1.12360238e-01
# CHECK-NEXT:   1.48292901e+00 -4.20779288e-01  1.40243283e-01 -7.03922624e-01
# CHECK-NEXT:   1.99264873e+00  2.03466877e+00  5.09549956e-01  2.82353429e-01
# CHECK-NEXT:  -8.76779854e-01 -4.32931491e-01  6.41354529e-01 -5.32490277e-01
# CHECK-NEXT:   2.42477064e+00  1.43649009e-01  5.60260754e-01 -2.66140365e-02
# CHECK-NEXT:   1.12960186e+00  2.82649856e+00 -7.65095146e-01  1.68513866e+00
# CHECK-NEXT:  -1.50632228e+00  1.59476497e+00 -9.37238534e-01  9.72722562e-02
# CHECK-NEXT:   2.81927278e-01  9.95346082e-01  9.54506434e-01 -1.04817880e+00
# CHECK-NEXT:   1.51420943e+00  1.38299166e+00  1.39864975e+00 -1.18470740e-01
# CHECK-NEXT:   7.09628837e-01  7.41448267e-01  2.41734989e+00  1.26382662e+00
# CHECK-NEXT:  -1.51470277e+00 -3.66052843e-01 -1.62153357e+00  1.80364879e+00
# CHECK-NEXT:   3.38518020e-01  7.44527251e-01 -4.92361617e-01  2.55702870e-01
# CHECK-NEXT:   6.26202963e-01 -1.02061824e+00  7.43938381e-01  3.23395433e-01
# CHECK-NEXT:  -8.37121192e-01  5.43333802e-01 -2.21502639e+00  1.52669123e+00
# CHECK-NEXT:   6.76366406e-02  3.24597115e-01  5.11502774e-01 -6.93189897e-01
# CHECK-NEXT:   5.89712742e-01  8.01211537e-01  4.53145629e-01 -1.89288406e+00
# CHECK-NEXT:   9.96279159e-01  6.05249020e-01  2.16391222e-01 -8.55512236e-01
# CHECK-NEXT:  -8.47976666e-01 -2.07358327e-01 -2.13439394e-01 -2.19382111e-02
# CHECK-NEXT:   1.06106380e+00  1.16222692e+00  2.20201125e+00  9.87167037e-01
# CHECK-NEXT:   7.36488419e-01 -1.03643701e+00  2.28836267e-01  1.37797143e+00
# CHECK-NEXT:   1.85533054e+00  8.74512864e-01  5.59156203e-01 -1.09394061e+00
# CHECK-NEXT:  -4.59887359e-01 -6.92883132e-01 -1.13131816e+00  3.88921022e-01
# CHECK-NEXT:  -1.24114851e+00 -2.00239181e+00  8.82661411e-01 -2.78313429e-01
# CHECK-NEXT:   4.58689117e-01  1.10159458e+00  1.30925142e+00 -1.37317449e+00
# CHECK-NEXT:   4.17446288e-02 -2.09853965e+00  4.11933130e-02 -1.43638446e+00
# CHECK-NEXT:   1.30947347e+00  1.72746032e+00 -6.56058792e-01 -8.89910535e-02]>: tensor<768xf32>} : () -> tensor<768xf32>
# CHECK-NEXT:   %2766 = "vm.call" @device.matmul(%2788,%3041) : (tensor<16x768xf32>,tensor<768x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2789 = "vm.call" @device.add(%2766,%3042) : (tensor<16x3072xf32>,tensor<3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2670 = "vm.const.f32" () {value = 3.141592653589793: f32} : () -> f32
# CHECK-NEXT:   %2671 = "vm.const.f32" () {value = 0.5: f32} : () -> f32
# CHECK-NEXT:   %2716 = "vm.call" @device.splat(%2671) : (f32) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2850 = "vm.call" @device.mul(%2716,%2789) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2672 = "vm.const.i32" () {value = 1: i32} : () -> i32
# CHECK-NEXT:   %2673 = "vm.const.i32" () {value = 2: i32} : () -> i32
# CHECK-NEXT:   %2697 = "vm.call" @device.div(%2673,%2670) : (i32,f32) -> (f32)
# CHECK-NEXT:   %2738 = "vm.call" @device.sqrt(%2697) : (f32) -> (f32)
# CHECK-NEXT:   %2674 = "vm.const.f32" () {value = 0.044715: f32} : () -> f32
# CHECK-NEXT:   %2675 = "vm.const.i32" () {value = 3: i32} : () -> i32
# CHECK-NEXT:   %2717 = "vm.call" @device.splat(%2675) : (i32) -> (i32)
# CHECK-NEXT:   %2934 = "vm.call" @device.pow(%2789,%2717) : (tensor<16x3072xf32>,i32) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2718 = "vm.call" @device.splat(%2674) : (f32) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2851 = "vm.call" @device.mul(%2718,%2934) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2790 = "vm.call" @device.add(%2789,%2851) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2719 = "vm.call" @device.splat(%2738) : (f32) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2852 = "vm.call" @device.mul(%2719,%2790) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %3104 = "vm.call" @device.tanh(%2852) : (tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2720 = "vm.call" @device.splat(%2672) : (i32) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2791 = "vm.call" @device.add(%2720,%3104) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2853 = "vm.call" @device.mul(%2850,%2791) : (tensor<16x3072xf32>,tensor<16x3072xf32>) -> (tensor<16x3072xf32>)
# CHECK-NEXT:   %2767 = "vm.call" @device.matmul(%2853,%3043) : (tensor<16x3072xf32>,tensor<3072x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2792 = "vm.call" @device.add(%2767,%3044) : (tensor<16x768xf32>,tensor<768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2793 = "vm.call" @device.add(%2786,%2792) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2676 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2677 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2678 = "vm.const.f32" () {value = 1e-05: f32} : () -> f32
# CHECK-NEXT:   %2953 = "vm.call" @device.reduce_sum(%2793) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3045 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2974 = "vm.call" @device.cast(%3045) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3150 = "vm.call" @device.div(%2953,%2974) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2954 = "vm.call" @device.reduce_sum(%2793) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3046 = "vm.rodata" () {value = dense<768>: tensor<1xi32>} : () -> tensor<1xi32>
# CHECK-NEXT:   %2975 = "vm.call" @device.cast(%3046) : (tensor<1xi32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3152 = "vm.call" @device.div(%2954,%2975) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2976 = "vm.call" @device.cast(%3152) : (tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2812 = "vm.call" @device.sub(%2793,%2976) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2856 = "vm.call" @device.mul(%2812,%2812) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2955 = "vm.call" @device.reduce_sum(%2793) : (tensor<16x768xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3151 = "vm.call" @device.div(%2955,%2975) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2813 = "vm.call" @device.sub(%2793,%3150) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2721 = "vm.call" @device.splat(%2677) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2858 = "vm.call" @device.mul(%2721,%2813) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2722 = "vm.call" @device.splat(%2676) : (f32) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2794 = "vm.call" @device.add(%3151,%2722) : (tensor<16x1xf32>,tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %2739 = "vm.call" @device.sqrt(%2794) : (tensor<16x1xf32>) -> (tensor<16x1xf32>)
# CHECK-NEXT:   %3153 = "vm.call" @device.div(%2858,%2739) : (tensor<16x768xf32>,tensor<16x1xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2723 = "vm.call" @device.splat(%2678) : (f32) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %2795 = "vm.call" @device.add(%3153,%2723) : (tensor<16x768xf32>,tensor<16x768xf32>) -> (tensor<16x768xf32>)
# CHECK-NEXT:   %3047 = "vm.rodata" () {value = dense<[1, 0]>: tensor<2xi32>} : () -> tensor<2xi32>
# CHECK-NEXT:   %3117 = "vm.call" @device.transpose(%2979,%3047) : (tensor<50257x768xf32>,tensor<2xi32>) -> (tensor<768x50257xf32>)
# CHECK-NEXT:   %2768 = "vm.call" @device.matmul(%2795,%3117) : (tensor<16x768xf32>,tensor<768x50257xf32>) -> (tensor<16x50257xf32>)
# CHECK-NEXT:   "vm.call" @print(%2768) : (tensor<16x50257xf32>) -> ()
# CHECK-NEXT: }) {target_info = {target_backend = sycl}} : () -> ()
