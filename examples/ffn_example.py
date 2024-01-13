import numpy as np

def gelu(x):
    pi = 3.141592653589793
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps = 0.00001):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b):
    return linear(gelu(linear(x, c_fc_w, c_fc_b)), c_proj_w, c_proj_b)

n_embed = 1600
n_head = 25
n_layers = 48
bs = 32

np.random.seed(42)
x = np.random.randn(bs, n_embed)
c_fc_w = np.random.randn(n_embed, 4 * n_embed)
c_fc_b = np.random.randn(4 * n_embed)
c_proj_w = np.random.randn(4 * n_embed, n_embed)
c_proj_b = np.random.randn(n_embed)

y = ffn(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b)

print(y)
