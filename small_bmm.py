import torch

import triton
from triton import language as tl

from time import time

@triton.jit
def small_bmm_kernel(
    a_ptr,
    b_ptr,
    o_ptr,
    a_rows: tl.constexpr,
    a_cols: tl.constexpr,
    b_rows: tl.constexpr,
    b_cols: tl.constexpr,
    out_dtype: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    
    # Build [a_rows, a_cols] matrix of pointers to read elements of a
    # [      0,          1,          2, ..., a_cols - 1 ]
    # [ a_cols, a_cols + 1, a_cols + 2, ..., 2 * a_cols - 1 ]
    # [ ... ]
    a_ptrs = a_ptr + (
        batch_idx * a_rows * a_cols + # Skip to the element in the batch this program / threadblock will be responsible for
        tl.arange(0, a_rows)[:, None] * a_cols + # [0, w, 2 * w, 3 * w, ... ], shape [a_rows, 1]
        tl.arange(0, a_cols)[None, :] # [0, 1, 2, 3, ... ], shape [1, a_cols]
    )

    # Build [b_rows, b_cols] matrix of pointers to read elements of b
    b_ptrs = b_ptr + batch_idx * b_rows * b_cols + tl.arange(0, b_rows)[:, None] * b_cols + tl.arange(0, b_cols)[None, :]
    )

    # Build [a_rows, b_cols] matrix of pointers to write output
    o_ptrs = o_ptr + (
        batch_idx * a_rows * b_cols +
        tl.arange(0, a_rows)[:, None] * b_cols +
        tl.arange(0, b_cols)[None, :]
    )

    # Actual algorithm!

    # Load elements of A and B to SRAM
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    # Matrix multiply, need to cast back to FP16 because by default accumulation happens in fp32
    o = tl.dot(a, b).to(out_dtype)

    # Store output
    tl.store(o_ptrs, o)


def triton_small_bmm(a, b):
    o = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype, device=a.device)

    # Launch one program per pair of matrices (1 for each batch element)
    grid = (a.shape[0],)

    small_bmm_kernel[grid](
        a, b, o,
        a_rows=a.shape[1], a_cols=a.shape[2],
        b_rows=b.shape[1], b_cols=b.shape[2],
        out_dtype=tl.float16,
        num_stages=4, # Tell triton to unroll internally generated loops by a factor of 4
        num_warps=8, # 8 warps / 256 threads per program
    )

    return o





# Test / Profile

batch_size = 16384

a = torch.randn((16384, 256, 128), dtype=torch.float16, device=torch.device('cuda:0'))
b = torch.randn((16384, 128, 128), dtype=torch.float16, device=torch.device('cuda:0'))

triton_out = triton_small_bmm(a, b)
torch_out = torch.bmm(a, b)

print(triton_out)
print(torch_out)

num_iters = 1000

with torch.no_grad():
    torch.cuda.synchronize()

    start = time()
    for i in range(num_iters):
        torch.bmm(a, b)

    torch.cuda.synchronize()
    end = time()

    diff = end - start

    print("Torch ", batch_size * num_iters / diff)


with torch.no_grad():
    torch.cuda.synchronize()

    start = time()
    for i in range(num_iters):
        triton_small_bmm(a, b)

    torch.cuda.synchronize()
    end = time()

    diff = end - start

    print("Triton", batch_size * num_iters / diff)
