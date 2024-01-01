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

# def attention(q, k, v, mask):
#     return softmax(q @ np.transpose(k) / np.sqrt(q.shape[-1]) + mask) @ v

# def mha(x, c_attn, c_proj, n_head):
#     x = linear(x, **c_attn)
#     qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
#     casual_mask = (1 - np.tri(x.shape[0])) * -1e10
#     out_heads = [attention(q, k, v, casual_mask) for q, k, v in zip(*qkv_heads)]
#     x = linear(np.hstack(out_heads), **c_proj)
#     return x

# def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
#     x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
#     x = x + ffn(layer_norm(x, **ln_2), **mlp)
#     return x

# def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
#     x = wte[inputs] + wpe[range(len(inputs))]
#     for block in blocks:
#         x = transformer_block(x, **block, n_head=n_head)
#     return layer_norm(x, **ln_f) @ wte.T

# def generate(inputs, params, n_head, n_tokens_to_generate):
#     from tqdm import tqdm
#     for _ in tqdm(range(n_tokens_to_generate), "generating"):
#         logits = gpt2(inputs, **params, n_head=n_head)
#         next_id = np.argmax(logits[-1])
#         inputs = np.append(inputs, [next_id])
#     return list(inputs[len(inputs) - n_tokens_to_generate :])

# def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
#     from utils import load_encoder_hparams_and_params
#     encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
#     input_ids = encoder.encode(prompt)
#     assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
#     output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
#     output_text = encoder.decode(output_ids)
#     return output_text