import torch

import triton
import triton.language as tl


@triton.jit
def _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols,
                  use_mask: tl.constexpr, use_bias: tl.constexpr):

    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)

    if use_bias:
        bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
        row += bias

    if use_mask:
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float("-inf")).to(tl.float32)
        row = tl.where(mask == 0, float("-1e20"), row)

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols,
                       is_bf16: tl.constexpr):
    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float(0))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float(0))

    if is_bf16:
        output_row = output_row.to(tl.float32)
        d_output_row = d_output_row.to(tl.float32)

    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row

    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_bias_kernel(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride,
                             output_row_stride, n_cols, n_heads, BLOCK_SIZE: tl.constexpr,
                             use_mask: tl.constexpr, use_bias: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    input_ptrs = input_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets

    mask_ptrs = input_ptrs  # place holder, not use if use_mask == False
    if use_mask:
        mask_row_ptr = mask_ptr + (row_idx // (n_heads * n_cols)) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets

    bias_ptrs = input_ptrs  # place holder, not use if use_bias == False
    if use_bias:
        bias_row_ptr = bias_ptr + (row_idx % (n_heads * n_cols)) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets

    _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask,
                  use_bias)


@triton.jit
def softmax_mask_bias_kernel_two_rows(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride,
                                      output_row_stride, n_cols, n_heads, BLOCK_SIZE: tl.constexpr,
                                      use_mask: tl.constexpr, use_bias: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    input_row_ptr = input_ptr + 2 * row_idx * input_row_stride
    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride

    input_ptrs = input_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets

    mask_ptrs = input_ptrs  # place holder, not use if use_mask == False
    if use_mask:
        mask_row_ptr = mask_ptr + ((2 * row_idx) // (n_heads * n_cols)) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets

    bias_ptrs = input_ptrs  # place holder, not use if use_bias == False
    if use_bias:
        bias_row_ptr = bias_ptr + ((2 * row_idx) % (n_heads * n_cols)) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets

    _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask,
                  use_bias)

    mask_ptrs = input_ptrs  # place holder, not use if use_mask == False
    if use_mask:
        mask_row_ptr = mask_ptr + ((2 * row_idx + 1) // (n_heads * n_cols)) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets

    bias_ptrs = input_ptrs  # place holder, not use if use_bias == False
    if use_bias:
        bias_row_ptr = bias_ptr + ((2 * row_idx + 1) % (n_heads * n_cols)) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets

    _softmax_core(input_ptrs + n_cols, output_ptrs + n_cols, mask_ptrs, bias_ptrs, col_offsets,
                  n_cols, use_mask, use_bias)


@triton.jit
def softmax_grad_kernel(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride,
                        output_row_stride, d_input_row_stride, n_cols, BLOCK_SIZE: tl.constexpr,
                        is_bf16: tl.constexpr):

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    output_row_ptr = output_ptr + row_idx * output_row_stride
    d_output_row_ptr = d_output_ptr + row_idx * d_output_row_stride
    d_input_row_ptr = d_input_ptr + row_idx * d_input_row_stride

    output_ptrs = output_row_ptr + col_offsets
    d_output_ptrs = d_output_row_ptr + col_offsets
    d_input_ptrs = d_input_row_ptr + col_offsets

    _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols, is_bf16)


@triton.jit
def softmax_grad_kernel_two_rows(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride,
                                      output_row_stride, d_input_row_stride, n_cols,
                                      BLOCK_SIZE: tl.constexpr, is_bf16: tl.constexpr):

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride
    d_output_row_ptr = d_output_ptr + 2 * row_idx * d_output_row_stride
    d_input_row_ptr = d_input_ptr + 2 * row_idx * d_input_row_stride

    output_ptrs = output_row_ptr + col_offsets
    d_output_ptrs = d_output_row_ptr + col_offsets
    d_input_ptrs = d_input_row_ptr + col_offsets

    _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols, is_bf16)

    _softmax_grad_core(output_ptrs + n_cols, d_output_ptrs + n_cols, d_input_ptrs + n_cols,
                       col_offsets, n_cols, is_bf16)


def softmax_triton_kernel_wrapper(x, mask, bias, n_rows, n_cols):
    y = torch.empty_like(x)
    n_heads = x.shape[2]

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _dispatch_kernel = softmax_mask_bias_kernel
    _grid = (n_rows,)
    if n_cols <= 128 and n_rows % 2 == 0:
        _dispatch_kernel = softmax_mask_bias_kernel_two_rows
        _grid = (n_rows // 2,)

    _dispatch_kernel[_grid](
        y,
        x,
        mask,
        bias,
        x.stride(-2),
        y.stride(-2),
        n_cols,
        n_heads,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
        use_mask=(mask != None),
        use_bias=(bias != None),
    )
    return y


def softmax_grad_triton_kernel_wrapper(grad_output, output, n_rows, n_cols):
    grad_input = torch.empty_like(grad_output)

    num_warps = 1
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE >= 1024:
        num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    is_bf16 = (output.dtype == torch.bfloat16)

    _dispatch_kernel = softmax_grad_kernel
    _grid = (n_rows,)
    if n_cols <= 128 and n_rows % 2 == 0:
        _dispatch_kernel = softmax_grad_kernel_two_rows
        _grid = (n_rows // 2,)

    _dispatch_kernel[_grid](
        grad_output,
        output,
        grad_input,
        grad_output.stride(-2),
        output.stride(-2),
        grad_output.stride(-2),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
        is_bf16=is_bf16,
    )
    return grad_input
