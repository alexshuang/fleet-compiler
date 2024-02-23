import numpy as np

def gelu(x):
    pi = 3.141592653589793
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, eps = 0.00001):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + eps) + beta

def linear(x, w, b):
    return x @ w + b

def ffn(x):
    return linear(gelu(linear(x, c_fc_w, c_fc_b)), c_proj_w, c_proj_b)

def attention(q, k, v, mask):
    kt = np.transpose(k)
    qkt = q @ kt
    scaled_qkt = softmax(qkt / np.sqrt(n_embed_per_head) + mask)
    return scaled_qkt @ v

def mha(x, n_head):
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
    # multi-head casual self attention
    x = x + mha(layer_norm(x), n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # position-wise feed forward network
    x = x + ffn(layer_norm(x))  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x

def gpt2(x):
    x = wte[inputs]
    x = x + wpe[pos_array]
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

# gpt2-small config
vocab_size = 50257
n_positions = 1024
n_embed = 768
n_head = 12
n_layers = 12
gamma = 0.00001
beta = 0.00001
n_embed_per_head = n_embed / n_head

bs = ctx_len = 16
inputs = [8897, 33125, 34028, 11310, 46496, 8936, 13422, 12673, 12521, 4655, 27264, 42624, 48419, 27095, 24398, 15221]
pos_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

np.random.seed(42)
wte = np.random.randn(vocab_size, n_embed)
wpe = np.random.randn(n_positions, n_embed)

# layer 0
c_fc_w = np.random.randn(n_embed, 4 * n_embed)
c_fc_b = np.random.randn(4 * n_embed)
c_proj_w = np.random.randn(4 * n_embed, n_embed)
c_proj_b = np.random.randn(n_embed)
c_attn_w = np.random.randn(n_embed, 3 * n_embed)
c_attn_b = np.random.randn(3 * n_embed)
c_attn_proj_w = np.random.randn(n_embed, n_embed)
c_attn_proj_b = np.random.randn(n_embed)

y = gpt2(inputs)
print(y)
