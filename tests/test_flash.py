import torch
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.bert_padding import unpad_input


def _flash_attn(q, k, v, mask, bias, window_size=(-1, -1)):
    """
    Args:
        q:
            (*, seqlen, nheads, headdim) query tensor
        k:
            (*, seqlen, nheads_k, headdim) key tensor
        v:
            (*, seqlen, nheads_k, headdim) value tensor
        mask:
            (*, seqlen), bool / int, 1 means valid and 0 means not valid.
        bias:
            (*, n_heads, seq_len, seq_len) bias tensor
        window_size:
            (left, right). If not (-1, -1), implements sliding window local attention.
    """
    batch_dims = q.shape[:-3]
    n, no_heads, c = q.shape[-3:]
    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()
    bias = bias.half()
    mask = mask.half()

    # [*, B, N, H, C]
    # q = q.transpose(-2, -3)
    # k = k.transpose(-2, -3)
    # v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]

    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])

    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    k_unpad, _, k_cu_seqlens, k_max_s = unpad_input(k, mask)
    v_unpad, _, v_cu_seqlens, v_max_s = unpad_input(v, mask)

    out = flash_attn_unpadded_func(
        q,
        k_unpad,
        v_unpad,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        attn_mask=mask,
        attn_bias=bias,
        dropout_p=0.,
        softmax_scale=1.,  # q has been scaled already
        window_size=window_size,
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c)

    out = out.to(dtype=dtype)

    return out


if __name__ == "__main__":
    bs = 8
    seqlen = 128
    n_heads = 2
    headdim = 16
    q = torch.randn(bs, seqlen, n_heads, headdim)
    k = torch.randn(bs, seqlen, n_heads, headdim)
    v = torch.randn(bs, seqlen, n_heads, headdim)
    bias = torch.randn(bs, n_heads, seqlen, seqlen, headdim)
    mask = torch.randint(0, 2, (bs, seqlen))

    output = _flash_attn(q, k, v, mask, bias)
    assert output.shape == (bs, seqlen, n_heads, headdim)
