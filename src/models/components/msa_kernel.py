# Original Author: Alex Zhang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import triton
import triton.language as tl

INF: tl.constexpr = 1e9

def nearest_pow2(n: int):
    power = math.ceil(math.log2(n))
    next_power_of_two = 2 ** power
    return next_power_of_two

@triton.jit
def MSAFwdFused(
    v_si_ptr, b_ij_ptr, g_si_ptr, output_ptr, vw_ptr,
    logsumexp_ptr,
    C_hidden, N_head,
    C_LEN_POW2: tl.constexpr,
    RES_LEN_POW2: tl.constexpr,
    SEQ_LEN: tl.constexpr, RES_LEN: tl.constexpr, 
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Compute the program ID and starting index
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    
    z_off = pid_z.to(tl.int64)
    h_off = pid_h.to(tl.int64)
    i_off = pid_i.to(tl.int64) * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    offs_c = tl.arange(0, C_LEN_POW2)
    
    # Use exp2 for Triton
    log2_e = 1.44269504089

    prev_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    new_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    l = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        # Load in b weight i:i+BLOCK_SIZE_ROW and compute softmax
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = (z_off * RES_LEN * RES_LEN * N_head) + \
                (offs_i[:, None] * RES_LEN * N_head) + \
                (offs_j[None, :] * N_head) + \
                (h_off)

        ij_mask = ((offs_i < RES_LEN)[:, None]) & ((offs_j < RES_LEN)[None, :])
        
        # Load current b's
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        
        # Compute softmax statistics (to broadcast)
        new_row_max = tl.maximum(tl.max(b, axis=1, keep_dims=True), prev_row_max)
        
        w = tl.exp2(log2_e * (b - new_row_max))
        l *= tl.exp2(log2_e * (prev_row_max - new_row_max))
        l += tl.sum(w, axis=1, keep_dims=True)
    
        # Compute vw portion
        for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
            # Offsets for {s,i} indices
            for ch in range(0, C_hidden, 1):
                offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
                si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                     (offs_s[None, :] * RES_LEN * N_head * C_hidden) + \
                     (offs_i[:, None] * N_head * C_hidden) + \
                     (h_off * C_hidden) + \
                     (ch)
                sj_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                         (offs_s[None, :] * RES_LEN * N_head * C_hidden) + \
                         (offs_j[:, None] * N_head * C_hidden) + \
                         (h_off * C_hidden) + \
                         (ch)
                si_mask = ((offs_s < SEQ_LEN)[None, :]) & ((offs_i < RES_LEN)[:, None])
                sj_mask = ((offs_s < SEQ_LEN)[None, :]) & ((offs_j < RES_LEN)[:, None])

                # Load in v_{s,j} transposed
                v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
                
                # Pay with extra loads of the outputs
                vw = tl.load(output_ptr + si_off, si_mask, 0)
                vw = vw * (tl.exp2(log2_e * (prev_row_max - new_row_max)))
                
                # (I x J) x (J x S) = I x S
                vw = tl.dot(w, v, acc=vw)
                
                # Store vw in output 
                tl.store(output_ptr + si_off, vw, si_mask)
        prev_row_max = new_row_max
        
    # Compute outputs after vw softmax is handled
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        for ch in range(0, C_hidden, 1):
            offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
            si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                     (offs_s[None, :] * RES_LEN * N_head * C_hidden) + \
                     (offs_i[:, None] * N_head * C_hidden) + \
                     (h_off * C_hidden) + \
                     (ch)
            si_mask = ((offs_s < SEQ_LEN)[None, :]) & ((offs_i < RES_LEN)[:, None])

            # Load in g_{s,i} transposed
            g = tl.load(g_si_ptr + si_off, si_mask, 0)
            g = tl.sigmoid(g)

            # vw is currently in out in memory.
            vw = tl.load(output_ptr + si_off, si_mask, 0)
            vw = vw / l
            
            out = g * vw
            tl.store(output_ptr + si_off, out, si_mask)
            
            # Store for backwards pass
            tl.store(vw_ptr + si_off, vw, si_mask)
    
    # Store logsumexp
    lse_off = (z_off * RES_LEN * N_head) + \
            (offs_i[:, None] * N_head) + \
            (h_off)

    lse_mask = (offs_i < RES_LEN)[:, None]
    tl.store(logsumexp_ptr + lse_off, (new_row_max + tl.log(l)), lse_mask)

@triton.jit
def MSABwdFused(
    b_ij_ptr, logsumexp_ptr,
    N_head,
    RES_LEN: tl.constexpr, 
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    # TODO: Update to fuse more operations more efficiently 
    
    # Compute the program ID and starting index
    pid_zh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_z = pid_zh // N_head
    pid_h = pid_zh % N_head
    
    # Use exp2 for Triton
    log2_e = 1.44269504089
    
    z_off = pid_z.to(tl.int64)
    h_off = pid_h.to(tl.int64)
    i_off = pid_i.to(tl.int64) * BLOCK_SIZE_ROW
    
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    
    # Load logsumexp
    lse_off = (z_off * RES_LEN * N_head) + \
            (offs_i[:, None] * N_head) + \
            (h_off)

    lse_mask = (offs_i < RES_LEN)[:, None]
    logsumexp = tl.load(logsumexp_ptr + lse_off, lse_mask, 0)
    
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = (z_off * RES_LEN * RES_LEN * N_head) + \
                (offs_i[:, None] * RES_LEN * N_head) + \
                (offs_j[None, :] * N_head) + \
                (h_off)

        ij_mask = ((offs_i < RES_LEN)[:, None]) & ((offs_j < RES_LEN)[None, :])

        # Load current b's
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        b = tl.exp2(log2_e * (b - logsumexp))
        tl.store(b_ij_ptr + b_offs, b, ij_mask)
        
    

class _MSAWeightedAveragingFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, g):
        """
        Fuse the softmax and linear combination step of MSA.
        """
        n_batches, n_seq, n_res, no_heads, c_hidden = v.shape
        
        # allocate output
        out = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden), device=g.device, dtype=g.dtype)
        vw = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden), device=g.device, dtype=g.dtype)
        logsumexp = torch.empty((n_batches, n_res, 1, no_heads), device=g.device, dtype=g.dtype)

        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 16
        BLOCK_SIZE_SEQ = 16
        n_res_pow2 = nearest_pow2(n_res)
        c_hidden_pow2 = nearest_pow2(c_hidden)

        grid = (
            n_batches,
            no_heads,
            triton.cdiv(n_res, BLOCK_SIZE_ROW),
        )
        
        MSAFwdFused[grid](
            v, b, g, out, vw, logsumexp,
            c_hidden, no_heads,
            c_hidden_pow2, n_res_pow2, 
            n_seq, n_res,
            BLOCK_SIZE_ROW,
            BLOCK_SIZE_SEQ,
            BLOCK_SIZE_COL,
        )
        
        ctx.save_for_backward(vw, v, b, g, logsumexp)
        ctx.n_batches = n_batches
        ctx.no_heads = no_heads
        ctx.n_seq = n_seq
        ctx.n_res = n_res
        ctx.c_hidden = c_hidden
        
        return out

    @staticmethod
    def backward(ctx, do):
        """
        TODO: Currently experiencing some precision issues with the Softmax, but computation
        otherwise is correct.
        """
        split_heads = nn.Unflatten(dim=-1, unflattened_size=(ctx.no_heads, ctx.c_hidden))
        vw, v, b, g, logsumexp = ctx.saved_tensors
        
        assert do.is_contiguous()
        
        # Can be tuned / different than fwd pass
        BLOCK_SIZE_ROW = 64
        BLOCK_SIZE_COL = 64
        BLOCK_SIZE_SEQ = 16
        n_res_pow2 = nearest_pow2(ctx.n_res)
        c_hidden_pow2 = nearest_pow2(ctx.c_hidden)
        
        grid = (
            ctx.n_batches * ctx.no_heads,
            triton.cdiv(ctx.n_res, BLOCK_SIZE_ROW),
        )
        
        # For now, compute softmax(B) and write to B
        MSABwdFused[grid](
            b, logsumexp,
            ctx.no_heads, ctx.n_res,
            BLOCK_SIZE_ROW,
            BLOCK_SIZE_COL,
        )
        
        # dv_c = (do * g)_c @ W
        G = F.sigmoid(g)
        A = split_heads(do) * G 
        dv = torch.einsum('bsrhc,brRh->bsRhc', A, b)
        
        # db = (do * G)^T @ V
        C = torch.einsum('brshc,bsRhc->brRhc', torch.transpose(A, dim0=1, dim1=2), v)
        A_vwT = A * torch.einsum('bsrhc,brRh->bsRhc', v, torch.transpose(b, dim0=1, dim1=2))
        A_vwT = torch.sum(A_vwT, 1).unsqueeze(2)
        
        # C * D - E
        db = b * torch.sum((C - A_vwT), -1)
        
        # dg = G * (1 - G) * (do * vw)
        dg = G * (1 - G) * split_heads(do * vw)
        
        return dv, db, dg

MSAWeightedAveragingFused = _MSAWeightedAveragingFused.apply
