import torch
import math

class FlashattentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        qrows_num, krows_num = Q.shape[-2], K.shape[-2]
        d = Q.shape[-1]
        q_tile_size = 16
        k_tile_size = 16
        O = torch.empty_like(Q)
        L = torch.empty((*Q.shape[:-1], 1), dtype=Q.dtype, device=Q.device)

        for i in range(0, qrows_num, q_tile_size):
            qrow_start = i
            qrow_end = min(qrow_start + q_tile_size, qrows_num)
            q = Q[..., qrow_start:qrow_end, :]
            o, l= 0, 0
            m = torch.full(
                (*Q.shape[:-2], qrow_end - qrow_start, 1), 
                fill_value=-torch.inf, 
                dtype=Q.dtype, 
                device=Q.device
            )
            for j in range(0, krows_num, k_tile_size):
                krow_start = j
                krow_end = min(krow_start + k_tile_size, krows_num)
                k = K[..., krow_start:krow_end, :]
                v = V[..., krow_start:krow_end, :]

                s = (q @ k.transpose(-2, -1)) / math.sqrt(d) #(..., B_q, B_k)
                mx = torch.maximum(m, torch.max(s, dim=-1, keepdim=True).values) #(..., B_q, 1)
                
                p = torch.exp(s - mx) #(..., B_q, B_k)
                l = torch.exp(m - mx) * l + p.sum(dim=-1, keepdim=True) #(..., B_q, 1)
                o = torch.exp(m - mx) * o + p @ v #(..., B_q, d)
                m = mx #(..., B_q, 1)

            O[..., qrow_start:qrow_end, :] = o / l
            L[..., qrow_start:qrow_end, :] = m + torch.log(l)

        ctx.save_for_backward(L.squeeze(-1), Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError