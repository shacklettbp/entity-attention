import torch
from torch import nn
from torch.nn import functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from flash_attn.modules.mha import MHA as FlashMHA

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(0)

import triton
from triton import language as tl

from time import time
import sys

batch_size = 16384
num_embed_channels = 128
num_heads = 4

num_qkv_channels = num_embed_channels
num_out_channels = num_embed_channels

torch_dtype = torch.float16
triton_dtype = tl.float16

#torch_dtype = torch.float32
#triton_dtype = tl.float32

torch_dev = torch.device('cuda:0')
profile_backward = True

#fake_inputs = frozen_dict.freeze({
#    p'self': random.normal(input_rnds[0], (batch_size, 64), dtype=torch_dtype),
#    'teammates': random.normal(input_rnds[1], (batch_size, 5, 16), dtype=torch_dtype),
#    'opponents': random.normal(input_rnds[2], (batch_size, 6, 16), dtype=torch_dtype),
#})

fake_embed = torch.randn((batch_size, 16, num_embed_channels), dtype=torch_dtype).cuda()

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
def atomic_accumulate_block(v, base_ptr, idx, h, w):
    ptrs = (
        base_ptr + idx * h * w +
        tl.arange(0, h)[:, None] * w +
        tl.arange(0, w)[None, :]
    )

    tl.atomic_add(ptrs, v)

@triton.jit
def compute_softmax(inputs):
    z = inputs - tl.max(inputs, axis=-1, keep_dims=True)
    exp = tl.exp(z)

    return (exp / tl.sum(exp, axis=-1, keep_dims=True)).to(inputs.dtype)


# Forward pass

@triton.jit
def triton_short_sa_fwd_kernel(
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
    softmax_scale: tl.constexpr,
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

        Q_scaled = (Q * softmax_scale).to(x.dtype)

        QK = tl.dot(Q_scaled, tl.trans(K)).to(x.dtype)

        A = compute_softmax(QK)

        ## Load v weights
        w_v = load_block(w_v_ptr, head_idx, embed_dim, qvk_head_dim).to(x.dtype)

        V = tl.dot(x, w_v).to(x.dtype)
        H = tl.dot(A, V).to(x.dtype)

        w_o = load_block(w_o_ptr, head_idx, qvk_head_dim, out_dim).to(x.dtype)

        out = tl.dot(H, w_o, out_dtype=accumulator.dtype)
        accumulator += out

    # Save results
    store_block(accumulator.to(x.dtype), out_ptr, batch_elem, seq_len, out_dim)


@triton.jit
def triton_short_sa_fwd_faster_kernel(
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
    softmax_scale: tl.constexpr,
    elems_per_program: tl.constexpr,
):
    batch_base_idx = tl.program_id(0) * elems_per_program

    for head_idx in range(0, num_heads):
        # Load weights for qk projection
        w_q = load_block(w_q_ptr, head_idx, embed_dim, qvk_head_dim).to(triton_dtype)
        w_k = load_block(w_k_ptr, head_idx, embed_dim, qvk_head_dim).to(triton_dtype)

        for batch_offset in range(0, elems_per_program):
            batch_idx = batch_base_idx + batch_offset

            # Load (seq_len, embed_dim) block of input into SRAM
            x = load_block(x_ptr, batch_idx, seq_len, embed_dim).to(triton_dtype)

            # tl.dot outputs to fp32 by default, cannot accumulate to bf16, so must
            # cast back to bf16
            Q = tl.dot(x, w_q).to(x.dtype)
            K = tl.dot(x, w_k).to(x.dtype)

            Q_scaled = (Q * softmax_scale).to(x.dtype)

            QK = tl.dot(Q_scaled, tl.trans(K)).to(x.dtype)

            A = compute_softmax(QK)

            ## Load v weights
            w_v = load_block(w_v_ptr, head_idx, embed_dim, qvk_head_dim).to(x.dtype)

            V = tl.dot(x, w_v).to(x.dtype)
            H = tl.dot(A, V).to(x.dtype)

            w_o = load_block(w_o_ptr, head_idx, qvk_head_dim, out_dim).to(x.dtype)

            out = tl.dot(H, w_o)
            if head_idx > 0:
                # Need to load result from prior heads
                out += load_block(out_ptr, batch_idx, seq_len, out_dim).to(tl.float32)

            # Save results
            store_block(out.to(x.dtype), out_ptr, batch_idx, seq_len, out_dim)


@triton.jit
def compute_softmax_frac(inputs):
    z = inputs - tl.max(inputs, axis=-1, keep_dims=True)
    exp = tl.exp(z)

    denom = tl.sum(exp, axis=-1, keep_dims=True)
    return exp, denom

@triton.jit
def softmax_derivative(dO, softmax_numer, softmax_denom):
    softmax_numer = softmax_numer.to(dO.dtype)
    softmax_denom = softmax_denom.to(dO.dtype)

    m = (1 / (softmax_denom * softmax_denom)).to(dO.dtype)
    n = (dO * m).to(dO.dtype)
    o = (n * softmax_numer).to(dO.dtype)
    p = tl.sum(o, axis=-1, keep_dims=True)
    s = dO / softmax_denom
    v = s - p
    return (v * softmax_numer).to(dO.dtype)

# Backward pass

@triton.jit
def triton_short_sa_bwd_kernel(
    d_out_ptr, 
    x_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    w_o_ptr,
    d_x_ptr,
    d_w_q_ptr,
    d_w_k_ptr,
    d_w_v_ptr,
    d_w_o_ptr,
    num_heads: tl.constexpr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    embed_dim: tl.constexpr,
    qvk_head_dim: tl.constexpr,
    out_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    elems_per_program: tl.constexpr
):
    start_batch_elem = tl.program_id(0) * elems_per_program

    for head_idx in range(0, num_heads):
        w_q = load_block(w_q_ptr, head_idx, embed_dim, qvk_head_dim).to(triton_dtype)
        w_k = load_block(w_k_ptr, head_idx, embed_dim, qvk_head_dim).to(triton_dtype)
        w_v = load_block(w_v_ptr, head_idx, embed_dim, qvk_head_dim).to(triton_dtype)
        w_o = load_block(w_o_ptr, head_idx, qvk_head_dim, out_dim).to(triton_dtype)

        d_w_q_accum = tl.zeros(w_q.shape, dtype=tl.float16)
        d_w_k_accum = tl.zeros(w_k.shape, dtype=tl.float16)
        d_w_v_accum = tl.zeros(w_v.shape, dtype=tl.float16)
        d_w_o_accum = tl.zeros(w_o.shape, dtype=tl.float16)

        for batch_offset in range(0, elems_per_program):
            # Load (seq_len, embed_dim) block of input into SRAM
            batch_idx = start_batch_elem + batch_offset

            x = load_block(x_ptr, batch_idx, seq_len, embed_dim).to(triton_dtype)
            d_out = load_block(d_out_ptr, batch_idx, seq_len, out_dim).to(triton_dtype)

            # Accumulate gradient for input sequence here. Need to load in
            # current d_x since each head will accumulate new results into global memory
            if head_idx == 0:
                d_x_accum = tl.zeros(x.shape, dtype=triton_dtype)
            else:
                d_x_accum = load_block(d_x_ptr, batch_idx, seq_len, embed_dim).to(triton_dtype)

            Q = tl.dot(x, w_q).to(x.dtype)
            K = tl.dot(x, w_k).to(x.dtype)

            Q_scaled = (Q * softmax_scale).to(x.dtype)

            QK = tl.dot(Q_scaled, tl.trans(K)).to(x.dtype)

            softmax_numer, softmax_denom = compute_softmax_frac(QK)
            A = (softmax_numer / softmax_denom).to(x.dtype)

            V = tl.dot(x, w_v).to(x.dtype)
            H = tl.dot(A, V).to(x.dtype)

            d_w_o = tl.dot(tl.trans(H), d_out).to(x.dtype)

            # Accumulate d_w_o into SRAM
            d_w_o_accum += d_w_o
            
            d_H = tl.dot(d_out, tl.trans(w_o)).to(x.dtype)
            d_V = tl.dot(tl.trans(A), d_H).to(x.dtype)
            d_w_v = tl.dot(tl.trans(x), d_V).to(x.dtype)

            # Accumulate d_w_v into SRAM
            d_w_v_accum += d_w_v

            # Accumulate the xWv part of the gradient wrt to x into SRAM
            d_x_accum += tl.dot(d_V, tl.trans(w_v), out_dtype=d_x_accum.dtype)

            d_A = tl.dot(d_H, tl.trans(V)).to(x.dtype)
            d_QK = (softmax_derivative(d_A, softmax_numer, softmax_denom) * softmax_scale).to(x.dtype)

            d_Q = tl.dot(d_QK, K).to(x.dtype)
            d_w_q = tl.dot(tl.trans(x), d_Q).to(x.dtype)

            # Accumulate d_w_q into SRAM
            d_w_q_accum += d_w_q

            # Accumulate the xWq part of the gradient wrt to x
            d_x_accum += tl.dot(d_Q, tl.trans(w_q), out_dtype=d_x_accum.dtype)

            d_K = tl.dot(tl.trans(d_QK), Q).to(x.dtype)
            d_w_k = tl.dot(tl.trans(x), d_K).to(x.dtype)

            # Accumulate d_w_k into SRAM
            d_w_k_accum += d_w_k

            # Accumulate the xWk part of the gradient wrt to x
            d_x_accum += tl.dot(d_K, tl.trans(w_k), out_dtype=d_x_accum.dtype)

            # Save input gradient so far
            store_block(d_x_accum, d_x_ptr, batch_idx, seq_len, embed_dim)

        atomic_accumulate_block(d_w_q_accum,
            d_w_q_ptr, head_idx, embed_dim, qvk_head_dim)

        atomic_accumulate_block(d_w_k_accum,
            d_w_k_ptr, head_idx, embed_dim, qvk_head_dim)

        atomic_accumulate_block(d_w_v_accum,
            d_w_v_ptr, head_idx, embed_dim, qvk_head_dim)

        atomic_accumulate_block(d_w_o_accum,
            d_w_o_ptr, head_idx, qvk_head_dim, out_dim)

class TritonShortSeqSelfAttention(torch.autograd.Function):
    @staticmethod
    def compute_metadata(x, w_q, w_o):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        embed_dim = x.shape[2]
    
        num_heads = w_q.shape[0]
        qvk_head_dim = w_q.shape[-1]
        out_dim = w_o.shape[-1]
    
        softmax_scale = qvk_head_dim ** -0.5

        return (batch_size, seq_len, embed_dim, num_heads,
                qvk_head_dim, out_dim, softmax_scale)

    @staticmethod
    def forward(ctx, x, w_q, w_k, w_v, w_o):
        (batch_size, seq_len, embed_dim, num_heads,
         qvk_head_dim, out_dim, softmax_scale) = (
            TritonShortSeqSelfAttention.compute_metadata(x, w_q, w_o))

        grid = (batch_size,)
    
        out = torch.empty((*x.shape[0:2], w_o.shape[-1]), dtype=x.dtype, device=x.device)

        triton_short_sa_fwd_kernel[grid](
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
            softmax_scale,
            num_stages=1, # This controls loop unrolling
            num_warps=4,
        )

        ctx.save_for_backward(x, w_q, w_k, w_v, w_o)

        return out

    @staticmethod
    def backward(ctx, d_out):
        x, w_q, w_k, w_v, w_o = ctx.saved_tensors

        # Want 128 programs
        elems_per_program = x.shape[0] // 128
        assert x.shape[0] % 128 == 0

        (batch_size, seq_len, embed_dim, num_heads,
         qvk_head_dim, out_dim, softmax_scale) = (
            TritonShortSeqSelfAttention.compute_metadata(x, w_q, w_o))

        grid = (batch_size // elems_per_program,)

        d_x = torch.empty_like(x) # Needs to be zero'd out
        d_w_q = torch.empty_like(w_q)
        d_w_k = torch.empty_like(w_k)
        d_w_v = torch.empty_like(w_v)
        d_w_o = torch.empty_like(w_o)

        triton_short_sa_bwd_kernel[grid](
            d_out,
            x,
            w_q,
            w_k,
            w_v,
            w_o,
            d_x,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            num_heads,
            batch_size,
            seq_len,
            embed_dim,
            qvk_head_dim,
            out_dim,
            softmax_scale,
            elems_per_program,
            num_stages=4, # This controls loop unrolling
            num_warps=4,
        )

        return d_x, d_w_q, d_w_k, d_w_v, d_w_o

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

        out = TritonShortSeqSelfAttention.apply(x, self.w_q, self.w_k, self.w_v, self.w_o)

        #if pad_amount > 0:
        #    out = out[:, 0:seq_len, :]

        return out

torch_model = nn.MultiheadAttention(
    num_qkv_channels,
    num_heads,
    bias=False,
    batch_first=True,
)

torch_model = torch_model.cuda()

flash_model = FlashMHA(
    embed_dim=num_embed_channels,
    num_heads=num_heads,
    qkv_proj_bias=False,
    out_proj_bias=False,
    use_flash_attn=True,
    dtype=torch_dtype,
    device=torch_dev,
)

triton_model = TritonShortSeqSelfAttentionModule(
    num_heads=num_heads,
    embed_features=num_embed_channels,
    qkv_features=num_qkv_channels,
    out_features=num_out_channels,
)
triton_model = triton_model.cuda()

with torch.no_grad():
    def txfm_in_proj_weight(w):
        w = w.view(3, num_heads, num_qkv_channels // num_heads, num_embed_channels)
        return w.permute(0, 1, 3, 2)

    torch_in_proj_weights = txfm_in_proj_weight(torch_model.in_proj_weight)
    triton_model.w_q.copy_(torch_in_proj_weights[0])
    triton_model.w_k.copy_(torch_in_proj_weights[1])
    triton_model.w_v.copy_(torch_in_proj_weights[2])

    def txfm_out_proj_weight(w):
        w = w.view(num_embed_channels, num_heads, num_qkv_channels // num_heads)
        return w.permute(1, 2, 0)

    triton_model.w_o.copy_(txfm_out_proj_weight(torch_model.out_proj.weight))

    flash_model.Wqkv.weight.copy_(torch_model.in_proj_weight)
    flash_model.out_proj.weight.copy_(torch_model.out_proj.weight)

torch_model = torch_model.to(dtype=torch_dtype)
triton_model = triton_model.to(dtype=torch_dtype)

torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=1e-10)
flash_optimizer = torch.optim.SGD(flash_model.parameters(), lr=1e-10)
triton_optimizer = torch.optim.SGD(triton_model.parameters(), lr=1e-10)

loss_fn = torch.nn.MSELoss()

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())

with torch.cuda.stream(s):
    torch_optimizer.zero_grad(set_to_none=True)
    flash_optimizer.zero_grad(set_to_none=True)
    triton_optimizer.zero_grad(set_to_none=True)

    torch_out = torch_model(fake_embed, fake_embed, fake_embed, need_weights=False)
    torch_out = torch_out[0]
    torch_loss = loss_fn(torch_out, fake_embed)
    torch_loss.backward()
    torch_optimizer.step()

    #flash_out = flash_model(fake_embed, max_seqlen = fake_embed.shape[1])
    #flash_loss = loss_fn(flash_out, fake_embed)
    #flash_loss.backward()
    #flash_optimizer.step()

    triton_out = triton_model(fake_embed)
    triton_loss = loss_fn(triton_out, fake_embed)
    triton_loss.backward()
    triton_optimizer.step()

torch.cuda.current_stream().wait_stream(s)

#print()
#print("Torch")
#print(torch_out)
#print()
#print("Flash")
#print(flash_out)
#print()
#print("Triton")
#print(triton_out)

with open('/tmp/triton_fwd.ptx', 'w') as f:
    print(list(triton_short_sa_fwd_kernel.cache[0].values())[0].asm['ptx'], file=f)

#with open('/tmp/triton_bwd.ptx', 'w') as f:
#    print(list(triton_short_sa_bwd_kernel.cache[0].values())[0].asm['ptx'], file=f)

baseline_graph = torch.cuda.CUDAGraph()
torch_optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(baseline_graph):
    if profile_backward:
        out = torch_model(fake_embed, fake_embed, fake_embed, need_weights=False)
        loss = loss_fn(out[0], fake_embed)
        loss.backward()
        torch_optimizer.step()
    else:
        with torch.no_grad():
            out = torch_model(fake_embed, fake_embed, fake_embed,
                              need_weights=False)

#flash_graph = torch.cuda.CUDAGraph()
#flash_optimizer.zero_grad(set_to_none=True)
#with torch.cuda.graph(flash_graph):
#    if profile_backward:
#        out = flash_model(fake_embed)
#        loss = loss_fn(out, fake_embed)
#        loss.backward()
#        flash_optimizer.step()
#    else:
#        with torch.no_grad():
#            out = flash_model(fake_embed)

triton_graph = torch.cuda.CUDAGraph()
triton_optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(triton_graph):
    if profile_backward:
        out = triton_model(fake_embed)
        loss = loss_fn(out, fake_embed)
        loss.backward()
        triton_optimizer.step()
    else:
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

    torch_elapsed = end - start

    print("Torch  {:.3} (1.00x) {:.3}".format(torch_elapsed, batch_size * num_iters / torch_elapsed))


#with torch.no_grad():
#    torch.cuda.synchronize()
#
#    start = time()
#    for i in range(num_iters):
#        flash_graph.replay()
#
#    torch.cuda.synchronize()
#    end = time()
#
#    diff = end - start
#
#    print("Flash ", batch_size * num_iters / diff)

#with torch.no_grad():
#    torch.cuda.synchronize()
#
#    start = time()
#    for i in range(num_iters):
#        triton_graph.replay()
#
#    torch.cuda.synchronize()
#    end = time()
#
#    diff = end - start
#
#    print("Triton", batch_size * num_iters / diff)

class FakeContext:
    def __init__(self, model):
        self.saved_tensors = [fake_embed, model.w_q, model.w_k, model.w_v, model.w_o]

    def save_for_backward(self, *args):
        pass

fake_ctx = FakeContext(triton_model)

TritonShortSeqSelfAttention.forward(fake_ctx, fake_embed,
                                    triton_model.w_q, triton_model.w_k,
                                    triton_model.w_v, triton_model.w_o)

TritonShortSeqSelfAttention.backward(fake_ctx, fake_embed)

with torch.no_grad():
    torch.cuda.synchronize()

    start = time()
    for i in range(num_iters):
        TritonShortSeqSelfAttention.forward(fake_ctx, fake_embed,
                                            triton_model.w_q, triton_model.w_k,
                                            triton_model.w_v, triton_model.w_o)
        if profile_backward:
            TritonShortSeqSelfAttention.backward(fake_ctx, fake_embed)

    torch.cuda.synchronize()
    end = time()

    triton_elapsed = end - start

    print("Triton {:.3} ({:.3}x) {:.3}".format(triton_elapsed, torch_elapsed / triton_elapsed, batch_size * num_iters / triton_elapsed))
