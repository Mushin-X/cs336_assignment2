import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale, # = 1/sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D), 
        strides=(stride_qq, stride_qd), 
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0), 
        block_shape=(Q_TILE_SIZE, D), 
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,), 
        block_shape=(Q_TILE_SIZE,), 
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") #(q_tile_size, d)
    o = tl.full(
        (Q_TILE_SIZE, D),
        0.0,
        tl.float32
    )
    l = tl.full(
        (Q_TILE_SIZE, 1),
        0.0,
        tl.float32
    )
    m = tl.full(
        (Q_TILE_SIZE, 1), 
        -float("inf"),
        tl.float32
    )
    for i in range(0, N_KEYS, K_TILE_SIZE):
        #(k_tile_size, d)
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s = tl.dot(q, tl.trans(k)) * scale #(q_tile_size, k_tile_size)

        if IS_CAUSAL:
            r = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            c = tl.arange(0, K_TILE_SIZE) + i

            mask = r[:,None] >= c[None,:]
            s = tl.where(mask, s, -float("inf"))

        mx = tl.maximum(m, tl.max(s, axis=-1, keep_dims=True)) #(q_tile_size, 1)

        p = tl.exp(s - mx) #(q_tile_size, k_tile_size)
        l = tl.exp(m - mx) * l + tl.sum(p, axis=-1, keep_dims=True) #(q_tile_size, 1)
        o = tl.exp(m - mx) * o + tl.dot(p, v) #(q_tile_size, d)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        m=mx

    tl.store(O_block_ptr, o / l, boundary_check=(0, 1))
    l = tl.reshape((m + tl.log(l)), (Q_TILE_SIZE,))
    tl.store(L_block_ptr, l, boundary_check=(0,))

class FlashattentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, qrows_num, d = Q.shape
        krows_num = K.shape[-2]
        q_tile_size, k_tile_size = 16, 16

        O = torch.empty((batch_size, qrows_num, d), dtype=Q.dtype, device=Q.device)
        L = torch.empty((batch_size, qrows_num), dtype=Q.dtype, device=Q.device)
         
        flash_fwd_kernel[(triton.cdiv(qrows_num, q_tile_size), batch_size)](
            Q_ptr=Q, K_ptr=K, V_ptr=V,
            O_ptr=O, L_ptr=L,
            stride_qb=qrows_num*d, stride_qq=d, stride_qd=1,
            stride_kb=krows_num*d, stride_kk=d, stride_kd=1,
            stride_vb=krows_num*d, stride_vk=d, stride_vd=1,
            stride_ob=qrows_num*d, stride_oq=d, stride_od=1,
            stride_lb=qrows_num, stride_lq=1,
            N_QUERIES=qrows_num, N_KEYS=krows_num,
            scale=1/math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            IS_CAUSAL=is_causal
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError