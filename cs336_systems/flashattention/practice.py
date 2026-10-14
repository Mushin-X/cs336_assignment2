import triton
import triton.language as tl
import torch

@triton.jit
def cal_sum(
    x_ptr, #输入指针
    output_ptr, #输出指针
    x_stride_row, x_stride_col, #输入x每移动一行或一列，分别需要越过多少个元素
    output_stride_row, #与上述相同
    num_row, num_col, #输入矩阵x的行数和列数
    row_tile_size: tl.constexpr, col_tile_size: tl.constexpr #
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(num_row, num_col),
        strides=(x_stride_row, x_stride_col),
        offsets=(row_tile_idx*row_tile_size, 0),
        block_shape=(row_tile_size, num_col),
        order=(1, 0)
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(num_row,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx*row_tile_size,),
        block_shape=(row_tile_size,),
        order=(0,)
    )
    output = tl.zeros((row_tile_size,), dtype=tl.float32)
    for _ in range(tl.cdiv(num_col, col_tile_size)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        output = tl.sum(row, axis=1)
        x_block_ptr = x_block_ptr.advance((0, col_tile_size))
    tl.store(output_block_ptr, output, boundary_check=(0,))

class SumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        num_row, num_col = x.shape[0], x.shape[1]
        ctx.save_for_backward(x)
        assert x.is_cuda, "x不在cuda设备上"
        assert x.is_contiguous(), "x张量不连续"
        ctx.row_tile_size = 2
        ctx.col_tile_size = 4
        ctx.input_shape = x.shape
        output = torch.empty(num_col, device=x.device)

        cal_sum[(tl.cdiv(num_row, ctx.row_tile_size),)](
            x_ptr=x,
            output_ptr=output,
            x_stride_row=num_col, x_stride_col=1,
            output_stride_row=1,
            num_row=num_row, num_col=num_col,
            row_tile_size=ctx.row_tile_size, col_tile_size=ctx.col_tile_size
        )
        return output

x = torch.randint(
    low=0,
    high=10,
    size=(4, 8),
    device='cuda'
)
sum1 = x.sum(dim=1)
sum2 = SumFunc(x)
print(sum1)
print(sum2)