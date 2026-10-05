# model/mamba_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_scan(a, b):
    """
    计算 h_t = a_t * h_{t-1} + b_t, h_{-1}=0。
    用倍增法并行：log2(L) 次迭代代替 L 次串行。
    a, b: [B, L, D, N]
    """
    B, L, D, N = a.shape
    A = a
    Bs = b
    step = 1
    while step < L:
        A_shift = torch.ones_like(A)
        A_shift[:, step:] = A[:, :-step].clone()
        B_shift = torch.zeros_like(Bs)
        B_shift[:, step:] = Bs[:, :-step].clone()

        Bs = A * B_shift + Bs
        A = A * A_shift
        step *= 2
    return Bs


class MambaBlock(nn.Module):
    """简化版选择性状态空间块（Mamba-like），并行扫描。"""

    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        self.ln = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        residual = x
        x = self.ln(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[..., :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        BC = self.x_proj(x_ssm)
        Bp, Cp = BC.chunk(2, dim=-1)

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(2)

        # 并行扫描：一次算完所有时间步的状态
        h_all = parallel_scan(dA, dB)                  # [B, L, d_inner, d_state]

        # 读出：y_t = sum_n h_t[n] * C_t[n]
        y = (h_all * Cp.unsqueeze(2)).sum(-1)          # [B, L, d_inner]

        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + self.dropout(y)