import jax
from jax import numpy as jnp

def compute_softmax(inputs):
    z = inputs - jnp.max(inputs, axis=-1, keepdims=True)

    exp = jnp.exp(z)
    return exp / jnp.sum(exp, axis=-1, dtype=jnp.float32, keepdims=True).astype(exp.dtype)

def compute_attention_head(x, w_q, w_k, w_v, w_o):
    Q = x @ w_q
    K = x @ w_k

    Q_scaled = Q / jnp.sqrt(K.shape[-1])

    QK = Q_scaled @ K.T

    A = compute_softmax(QK)

    V = x @ w_v
    H = A @ V

    return H @ w_o

def compute_partial_gradient(d_o, x, w_q, w_k, w_v, w_o):
    _, vjp_func = jax.vjp(compute_attention_head, x, w_q, w_k, w_v, w_o)
    return vjp_func(d_o)

make_jaxpr_fn = jax.make_jaxpr(compute_partial_gradient)

batch_size = 16384
num_channels = 128
seq_len = 16

fake_inputs = [
    jax.ShapeDtypeStruct(shape=(seq_len, num_channels), dtype=jnp.float16),
    jax.ShapeDtypeStruct(shape=(seq_len, num_channels), dtype=jnp.float16),
    jax.ShapeDtypeStruct(shape=(num_channels, num_channels), dtype=jnp.float16),
    jax.ShapeDtypeStruct(shape=(num_channels, num_channels), dtype=jnp.float16),
    jax.ShapeDtypeStruct(shape=(num_channels, num_channels), dtype=jnp.float16),
    jax.ShapeDtypeStruct(shape=(num_channels, num_channels), dtype=jnp.float16),
]

jaxpr = make_jaxpr_fn(*fake_inputs)

print("jaxpr:")
print(jaxpr)

input()
print("XLA:")

compiled = jax.jit(compute_partial_gradient).lower(*fake_inputs).compile()

print(compiled.as_text())
