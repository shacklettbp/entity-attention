import jax
from jax import lax, random, numpy as jnp
from jax.experimental import pallas as pl
import flax
from flax import linen as nn
from flax.core import FrozenDict, frozen_dict

from typing import List, Callable, Any
from functools import partial

import madrona_learn
from madrona_learn.models import EntitySelfAttentionNet, SelfAttention as FlaxSelfAttention

madrona_learn.init(0.6)

batch_size = 2
num_embed_channels = 64
num_out_channels = 64
num_heads = 1

float_dtype = jnp.float32

base_rnd = random.PRNGKey(5)
init_rnd, input_rnd = random.split(base_rnd)
input_rnds = random.split(input_rnd, 4)

#fake_inputs = frozen_dict.freeze({
#    'self': random.normal(input_rnds[0], (batch_size, 64), dtype=float_dtype),
#    'teammates': random.normal(input_rnds[1], (batch_size, 5, 16), dtype=float_dtype),
#    'opponents': random.normal(input_rnds[2], (batch_size, 6, 16), dtype=float_dtype),
#})

fake_embed = random.normal(input_rnds[3], (batch_size, 16, num_embed_channels), dtype=float_dtype)

def make_net(attention_cls, attention_only):
    if attention_only:
        model = attention_cls(
            num_heads = num_heads,
            qkv_features = num_embed_channels,
            out_features = num_out_channels,
            dtype = float_dtype,
        )
    else:
        model = EntitySelfAttentionNet(
            num_embed_channels = num_embed_channels,
            num_out_channels = num_out_channels,
            num_heads = num_heads,
            dtype = float_dtype,
            attention_cls = attention_cls,
        )

    @jax.jit
    def init():
        if attention_only:
            return model.init(init_rnd, fake_embed)
        else:
            return model.init(init_rnd, fake_inputs, train=True)
    
    params = init()

    print(jax.tree_map(jnp.shape, params))

    return model, params


def pallas_fused_short_sa_kernel(x_ref, w_qkv_ref, w_o_ref, out_ref):
    # Load input from DRAM. This will hopefully stay in SRAM while each head
    # performs attention
    x = x_ref[...]

    num_heads = w_qkv_ref.shape[0]

    def compute_softmax(inputs):
        z = inputs - jnp.max(inputs, axis=-1, keepdims=True)

        exp = jnp.exp(z)
        return exp / jnp.sum(exp, axis=-1, dtype=jnp.float32, keepdims=True).astype(exp.dtype)

    accumulator = jnp.zeros(out_ref.shape, dtype=jnp.float32)
    def single_head_attention(head_idx, accumulator):
        # Load weights for qk projection
        w_q = pl.load(w_qkv_ref, (head_idx, 0)).astype(x.dtype)
        w_k = pl.load(w_qkv_ref, (head_idx, 1)).astype(x.dtype)

        Q = x @ w_q
        K = x @ w_k

        QK = Q @ K.T

        softmax = compute_softmax(QK / jnp.sqrt(K.shape[-1]).astype(QK.dtype))

        ## Load v weights
        w_v = pl.load(w_qkv_ref, (head_idx, 2)).astype(x.dtype)
        V = x @ w_v

        attention = softmax @ V

        w_o = pl.load(w_o_ref, (head_idx,)).astype(x.dtype)
        out = attention @ w_o

        accumulator = accumulator + out.astype(accumulator.dtype)
        return accumulator

    accumulator = lax.fori_loop(
        0, num_heads, single_head_attention, accumulator)

    # Save results to DRAM
    out_ref[...] = accumulator.astype(out_ref.dtype)

def pallas_fused_short_sa_call(x, w_qkv, w_o, out_shape):
    x_spec = pl.BlockSpec(
        lambda i: (i, 0, 0), (None, x.shape[-2], x.shape[-1]))

    w_qkv_spec = pl.BlockSpec(
        lambda _: (0, 0, 0, 0), w_qkv.shape)

    w_o_spec = pl.BlockSpec(
        lambda _: (0, 0, 0), w_o.shape)

    o_spec = pl.BlockSpec(
        lambda i: (i, 0, 0), (None, out_shape[-2], out_shape[-1]))

    in_specs = [x_spec, w_qkv_spec, w_o_spec]

    return pl.pallas_call(
        pallas_fused_short_sa_kernel,
        grid = (batch_size),
        in_specs = in_specs,
        out_specs = o_spec,
        out_shape = jax.ShapeDtypeStruct(shape=out_shape, dtype=x.dtype),
    )(x, w_qkv, w_o)


class PallasShortSeqSelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        embedding_size = x.shape[-1]
        seq_len = x.shape[-2]

        assert seq_len <= 16
        pad_amount = 16 - seq_len 

        if pad_amount > 0:
            x = jnp.pad(x, ((0, 0), (0, pad_amount), (0, 0)),
                        constant_values = 0)

        w_qkv_shape = (
            self.num_heads,
            3,
            self.qkv_features,
            self.qkv_features // self.num_heads,
        )

        w_o_shape = (
            self.num_heads,
            self.qkv_features // self.num_heads,
            self.out_features,
        )

        w_qkv = self.param('w_qkv',
            lambda rng, shape: nn.initializers.lecun_normal()(
                rng, shape, jnp.float32), w_qkv_shape)
        
        w_o = self.param('w_o',
            lambda rng, shape: nn.initializers.lecun_normal()(
                rng, shape, jnp.float32), w_o_shape)

        out = pallas_fused_short_sa_call(x, w_qkv, w_o,
            (x.shape[0], seq_len + pad_amount, self.out_features))

        if pad_amount > 0:
            out = out[:, 0:seq_len, :]

        return out


flax_model, flax_params = make_net(FlaxSelfAttention, True)
print()
pallas_model, pallas_params = make_net(PallasShortSeqSelfAttention, True)

@jax.jit
def test_flax(params, *inputs):
    return flax_model.apply(params, *inputs)

@jax.jit
def test_pallas(params, *inputs):
    return pallas_model.apply(params, *inputs)

qvk_flax_params = jnp.stack([
        jnp.transpose(flax_params['params']['SelfAttention_0']['query']['kernel'], (1, 0, 2)),
        jnp.transpose(flax_params['params']['SelfAttention_0']['key']['kernel'], (1, 0, 2)),
        jnp.transpose(flax_params['params']['SelfAttention_0']['value']['kernel'], (1, 0, 2)),
    ], axis = 1)

pallas_params['params']['out_weights'] = \
    flax_params['params']['SelfAttention_0']['out']['kernel']

pallas_params['params']['qkv_weights'] = qvk_flax_params

flax_out = test_flax(flax_params, fake_embed)
pallas_out = test_pallas(pallas_params, fake_embed)

print(flax_out[0, 0])
print(pallas_out[0, 0])

print(pallas_out.shape)
