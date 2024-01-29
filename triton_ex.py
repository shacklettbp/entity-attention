import torch
from torch import nn
from torch.nn import functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(0)

import triton
from triton import language as tl
from time import time

batch_size = 16384
num_embed_channels = 128
num_qvk_channels = 128
num_out_channels = 128
num_heads = 4

float_dtype = torch.float16
triton_dtype = tl.float16

#float_dtype = torch.float32
#triton_dtype = tl.float32


#fakea_inputs = frozen_dict.freeze({
#    p'self': random.normal(input_rnds[0], (batch_size, 64), dtype=float_dtype),
#    'teammates': random.normal(input_rnds[1], (batch_size, 5, 16), dtype=float_dtype),
#    'opponents': random.normal(input_rnds[2], (batch_size, 6, 16), dtype=float_dtype),
#})

fake_embed = torch.randn((batch_size, 16, num_embed_channels), dtype=float_dtype).cuda()

@triton.jit
def load_block(base_ptr, idx, h, w):
    ptrs = (
        base_ptr + idx * h * w +
        tl.arange(0, h)[:, None] * w +
        tl.arange(0, w)[None, :]
    )

    return tl.load(ptrs)

@triton.jit
def store_block(v, base_ptr, idx, h, w):
    ptrs = (
        base_ptr + idx * h * w +
        tl.arange(0, h)[:, None] * w +
        tl.arange(0, w)[None, :]
    )

    tl.store(ptrs, v)

@triton.jit
def compute_softmax(inputs):
    z = inputs - tl.max(inputs, axis=-1, keep_dims=True)

    exp = tl.exp(z).to(inputs.dtype)
    return (exp / tl.sum(exp.to(tl.float32), axis=-1, keep_dims=True).to(exp.dtype)).to(exp.dtype)

@triton.jit
def triton_fused_short_sa_kernel(
    x_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    w_o_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    qvk_head_dim: tl.constexpr,
    out_dim: tl.constexpr,
    q_scale: tl.constexpr,
):
    batch_elem = tl.program_id(0)

    # Load (seq_len, embed_dim) block of input into SRAM
    x = load_block(x_ptr, batch_elem, seq_len, embed_dim).to(triton_dtype)

    accumulator = tl.zeros((seq_len, out_dim), dtype=tl.float32)

    for head_idx in range(0, num_heads):
        # Load weights for qk projection
        w_q = load_block(w_q_ptr, head_idx, embed_dim, qvk_head_dim).to(x.dtype)
        w_k = load_block(w_k_ptr, head_idx, embed_dim, qvk_head_dim).to(x.dtype)

        # tl.dot outputs to fp32 by default, cannot accumulate to bf16, so must
        # cast back to bf16
        Q = tl.dot(x, w_q).to(x.dtype)
        K = tl.dot(x, w_k).to(x.dtype)

        Q_scaled = (Q * q_scale).to(x.dtype)

        QK = tl.dot(Q_scaled, tl.trans(K)).to(x.dtype)

        softmax = compute_softmax(QK)

        ## Load v weights
        w_v = load_block(w_v_ptr, head_idx, embed_dim, qvk_head_dim).to(x.dtype)

        V = tl.dot(x, w_v).to(x.dtype)
        attention = tl.dot(softmax, V).to(x.dtype)

        w_o = load_block(w_o_ptr, head_idx, qvk_head_dim, out_dim).to(x.dtype)

        out = tl.dot(attention, w_o, out_dtype=accumulator.dtype)
        accumulator += out

    # Save results
    store_block(accumulator.to(x.dtype), out_ptr, batch_elem, seq_len, out_dim)

class TritonShortSeqSelfAttention(torch.autograd.Function):
    def forward(ctx, x, w_q, w_k, w_v, w_o, out):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]
    
        num_heads = w_q.shape[0]
        qvk_head_dim = w_q.shape[-1]
        out_dim = w_o.shape[-1]
    
        q_scale = qvk_head_dim ** -0.5
    
        grid = (batch_size,)
    
        triton_fused_short_sa_kernel[grid](
            x,
            w_q,
            w_k,
            w_v,
            w_o,
            out,
            num_heads,
            batch_size,
            seq_len,
            embed_dim,
            qvk_head_dim,
            out_dim,
            q_scale,
            num_stages=1, # This controls loop unrolling
            num_warps=4,
        )

        return out

class TritonShortSeqSelfAttentionModule(nn.Module):
    def __init__(self, num_heads, embed_features, qkv_features, out_features):
        super().__init__()

        w_qkv_shape = (
            num_heads,
            embed_features,
            qkv_features // num_heads,
        )

        w_o_shape = (
            num_heads,
            qkv_features // num_heads,
            out_features,
        )

        self.w_q = nn.Parameter(torch.randn(w_qkv_shape, dtype=torch.float32))
        self.w_k = nn.Parameter(torch.randn(w_qkv_shape, dtype=torch.float32))
        self.w_v = nn.Parameter(torch.randn(w_qkv_shape, dtype=torch.float32))
        self.w_o = nn.Parameter(torch.randn(w_o_shape, dtype=torch.float32))

    def forward(self, x):
        seq_len = x.shape[-2]

        #assert seq_len <= 16
        #pad_amount = 16 - seq_len 

        #if pad_amount > 0:
        #    x = F.pad(x, (0, 0, 0, pad_amount, 0, 0), value = 0)

        out = torch.empty((*x.shape[0:2], self.w_o.shape[-1]), dtype=x.dtype, device=x.device)

        TritonShortSeqSelfAttention.apply(x, self.w_q, self.w_k, self.w_v, self.w_o, out)

        #if pad_amount > 0:
        #    out = out[:, 0:seq_len, :]

        return out

torch_model = nn.MultiheadAttention(
    num_qvk_channels,
    num_heads,
    bias=False,
    batch_first=True,
)

torch_model = torch_model.cuda()
torch_model.eval()

triton_model = TritonShortSeqSelfAttentionModule(
    num_heads=num_heads,
    embed_features=num_embed_channels,
    qkv_features=num_qvk_channels,
    out_features=num_out_channels,
)
triton_model = triton_model.cuda()

with torch.no_grad():
    def txfm_in_proj_weight(w):
        w = w.view(3, num_heads, num_qvk_channels // num_heads, num_embed_channels)
        return w.permute(0, 1, 3, 2)

    torch_in_proj_weights = txfm_in_proj_weight(torch_model.in_proj_weight)
    triton_model.w_q.copy_(torch_in_proj_weights[0])
    triton_model.w_k.copy_(torch_in_proj_weights[1])
    triton_model.w_v.copy_(torch_in_proj_weights[2])

    def txfm_out_proj_weight(w):
        w = w.view(num_embed_channels, num_heads, num_qvk_channels // num_heads)
        return w.permute(1, 2, 0)

    triton_model.w_o.copy_(txfm_out_proj_weight(torch_model.out_proj.weight))

torch_model = torch_model.to(dtype=float_dtype)
triton_model = triton_model.to(dtype=float_dtype)

torch_out = torch_model(fake_embed, fake_embed, fake_embed, need_weights=False)
triton_out = triton_model(fake_embed)

print(torch_out)
print(triton_out)

with open('/tmp/triton.ptx', 'w') as f:
    print(list(triton_fused_short_sa_kernel.cache[0].values())[0].asm['ptx'], file=f)

baseline_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(baseline_graph):
    with torch.no_grad():
        torch_out = torch_model(fake_embed, fake_embed, fake_embed, need_weights=False)


triton_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(triton_graph):
    with torch.no_grad():
        triton_out = triton_model(fake_embed)

num_iters = 1000

with torch.no_grad():
    torch.cuda.synchronize()

    start = time()
    for i in range(num_iters):
        baseline_graph.replay()

    torch.cuda.synchronize()
    end = time()

    diff = end - start

    print("Torch ", batch_size * num_iters / diff)

with torch.no_grad():
    torch.cuda.synchronize()

    start = time()
    for i in range(num_iters):
        triton_graph.replay()
        #triton_model(fake_embed)

    torch.cuda.synchronize()
    end = time()

    diff = end - start

    print("Triton", batch_size * num_iters / diff)
