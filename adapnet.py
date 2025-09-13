import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =========================
# 工具：DropPath（Stochastic Depth）
# =========================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # 按 batch 采样，保持时间和通道不变
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


# =========================
#  S4 / Mamba 组件
# =========================

class S4Layer(nn.Module):
    """S4 (Structured State Space) Layer 
    进一步收紧 dt_max，状态演化更慢。
    """

    def __init__(self, d_model, d_state=16, dropout=0.15, transposed=True,
                 dt_min=1e-3, dt_max=0.02):  # ← 从 0.05 收紧到 0.02
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed

        # A 的稳定参数化：对正数取 log，用时再加负号
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.register_parameter("A_log", nn.Parameter(torch.log(A_init)))  # log(|A|)

        # B 和 C（小初始化）
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        # D（跳跃连接）
        self.D = nn.Parameter(torch.ones(d_model))

        # 时间步长参数（log-dt 方式）
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        """
        u: (B, L, H)
        """
        B, L, H = u.shape

        # dt 稳定参数化与裁剪
        log_dt = self.dt_proj(u)                       # (B, L, H)
        dt = torch.clamp(torch.exp(log_dt), self.dt_min, self.dt_max)

        # 连续 A：负实数，确保稳定
        A = -torch.exp(self.A_log)                     # (H, N)

        # ZOH 离散化（稳定数值实现）
        Z = torch.einsum('blh,hn->blhn', dt, A)        # (B, L, H, N)
        A_bar = torch.exp(Z)                           # (B, L, H, N) in (0,1)

        # ((exp(dt*A)-I)/A) = expm1(Z) / A
        expm1_Z = torch.expm1(Z)                       # e^Z - 1
        denom = A.unsqueeze(0).unsqueeze(0)            # (1, 1, H, N)
        coef = expm1_Z / denom                         # (B, L, H, N)

        # 输入注入项：coef * B * u
        B_expanded = self.B.unsqueeze(0).unsqueeze(0)  # (1, 1, H, N)
        Bu = coef * B_expanded                         # (B, L, H, N)
        Bu = Bu * u.unsqueeze(-1)                      # (B, L, H, N)

        y = self.selective_scan(A_bar, Bu, self.C, self.D, u)
        return self.dropout(y)

    def selective_scan(self, A_bar, Bu, C, D, u):
        """选择性扫描（稳定递推）"""
        B, L, H, N = A_bar.shape
        x = torch.zeros(B, H, N, device=u.device, dtype=u.dtype)
        outputs = []
        for i in range(L):
            # x_k = A_bar * x_{k-1} + Bu
            x = x * A_bar[:, i] + Bu[:, i]
            # y_k = C x_k + D u_k
            y = torch.einsum('bhn,hn->bh', x, C) + D * u[:, i]
            outputs.append(y)
        return torch.stack(outputs, dim=1)  # (B, L, H)


class MambaBlock(nn.Module):
    """完整的 Mamba 块（配稳定 S4 与零初始化 out_proj）"""

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # 输入投影（分两支，一支做门控）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 深度可分离卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # 稳定版 SSM
        self.ssm = S4Layer(self.d_inner, d_state, dropout=dropout)

        # 输出投影 —— 零初始化（SkipInit 友好）
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        x_and_res = self.in_proj(x)                    # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # 1D depthwise conv
        x = x.transpose(1, 2)                          # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]                   # 去 padding
        x = x.transpose(1, 2)                          # (B, L, d_inner)

        x = F.silu(x)
        x = self.ssm(x)                                # (B, L, d_inner)

        # 门控
        x = x * F.silu(res)

        out = self.out_proj(x)
        return self.dropout(out)


class CustomMamba(nn.Module):
    """自定义 Mamba（Pre-LN + 残差缩放/SkipInit + DropPath）"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_layers=1,
                 dropout=0.15, drop_path_rate=0.10):  # ← 新增 drop_path_rate
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Pre-LN + 残差缩放
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        # 将 layerscale 从 0.03 降到 0.01，进一步放慢有效更新
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(0.01)) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # 分层的 DropPath（线性递增）
        dpr = torch.linspace(0, drop_path_rate, steps=num_layers).tolist()
        self.drop_paths = nn.ModuleList([DropPath(p) for p in dpr])

    def forward(self, x):
        """
        x: (B, L, D)
        """
        for i, layer in enumerate(self.layers):
            y = layer(self.norms[i](x))               # Pre-LN
            y = self.drop_paths[i](y)                 # ← Stochastic Depth
            x = x + self.alphas[i] * y                # 残差缩放
        return self.final_norm(x)


# =========================
# （快速版）
# =========================

class FastSSM(nn.Module):
    """快速 SSM（主要用于轻量推理）"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner)

        # 简化的状态空间参数
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, self.d_inner) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        """快速前向传播"""
        B, L, D = x.shape

        x = F.silu(self.in_proj(x))  # (B, L, d_inner)

        h = torch.zeros(B, self.d_inner, self.A.shape[1], device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = h * 0.9 + torch.einsum('bi,ij->bij', x[:, t], self.B)  # 状态更新
            y = torch.einsum('bij,ji->bi', h, self.C) + self.D * x[:, t]  # 输出
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return self.out_proj(y)


# 统一接口
class MambaWrapper(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 implementation='full', num_layers=1, dropout=0.15, drop_path_rate=0.10):
        """
        implementation: 'full' 使用完整实现, 'fast' 使用快速实现
        """
        super().__init__()
        self.implementation = implementation

        if implementation == 'full':
            self.mamba = CustomMamba(d_model, d_state, d_conv, expand,
                                     num_layers=num_layers, dropout=dropout,
                                     drop_path_rate=drop_path_rate)
        else:
            self.mamba = FastSSM(d_model, d_state, d_conv, expand)

        print(f"Initialized {implementation} Mamba implementation with d_model={d_model}")

    def forward(self, x):
        return self.mamba(x)


# =========================
# 输出控制头：温度缩放 + L2 归一化
# =========================
class OutController(nn.Module):
    """
    - learnable temperature（初始较小）：降低前期特征范数 → 距离更温和 → 损失下降慢
    - 可选 L2 normalize：将度量限制在可控范围，避免前期快速“越界”
    """
    def __init__(self, dim, init_temp=0.15, normalize=True):
        super().__init__()
        # 用 sigmoid 保证(0,1)区间；将 init_temp 反推为 logit
        # temp = sigmoid(t)；init_temp in (0,1)
        eps = 1e-6
        init_temp = float(np.clip(init_temp, eps, 1 - eps))
        init_t = np.log(init_temp) - np.log(1 - init_temp)
        self.t = nn.Parameter(torch.tensor(init_t, dtype=torch.float32))
        self.normalize = normalize

    def forward(self, x):
        # 温度在 (0,1)
        temp = torch.sigmoid(self.t)
        y = x * temp
        if self.normalize:
            y = F.normalize(y, p=2, dim=-1)
        return y


# =========================
# adapnet
# =========================

class adapnet(nn.Module):
    """
    Conv1D + (stitched) Mamba -> Gated fusion -> Residual -> Temporal pool -> OutControl
    """
    def __init__(
        self,
        inDims,
        outDims,
        seqL,
        w=5,
        use_mamba=True,
        mamba_impl='full',
        num_mamba_layers=1,
        dropout=0.30,
        drop_path_rate=0.10,        # ← 与主干一致
        out_init_temp=0.15,         # ← 输出温度初值（越小越慢）
        out_l2norm=True             # ← L2 归一化（默认开）
    ):
        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.use_mamba = use_mamba

        # 1) 局部卷积（保长度）
        pad = (self.w - 1) // 2
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w, padding=pad, bias=True)

        # 2) Pre-LN
        self.ln = nn.LayerNorm(outDims)

        # 3) fused Mamba
        if self.use_mamba:
            self.mamba = MambaWrapper(
                d_model=outDims,
                d_state=16,
                d_conv=4,
                expand=2,
                implementation=mamba_impl,
                num_layers=num_mamba_layers,
                dropout=dropout,
                drop_path_rate=drop_path_rate
            )

        # 4) 门控与残差（更小的残差系数）
        self.gate_proj = nn.Linear(outDims, outDims)
        self.alpha = nn.Parameter(torch.tensor(0.01))  # ← 从 0.03 降到 0.01
        self.dropout = nn.Dropout(dropout)
        self.branch_drop = DropPath(drop_path_rate)    # 对融合分支再做一次 drop-path

        # 5) 输出控制（温度 + 归一化）
        self.out_ctrl = OutController(outDims, init_temp=out_init_temp, normalize=out_l2norm)

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, outDims]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,C] -> [B,1,C]

        # Conv: [B,T,C] -> [B,C,T] -> [B,outDims,T] -> [B,T,outDims]
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = x_conv.permute(0, 2, 1)  # [B, T, outDims]

        if not self.use_mamba or x_conv.size(1) <= 1:
            y = torch.mean(x_conv, dim=1)            # 退化：Conv + 池化
            return self.out_ctrl(y)                  # 输出侧控制

        # Pre-LN
        x_norm = self.ln(x_conv)

        # Stitched Mamba
        x_mamba = self.mamba(x_norm)                 # [B, T, outDims]

        # 门控融合
        gate = F.silu(self.gate_proj(x_conv))
        fused = x_mamba * gate
        fused = self.dropout(fused)
        fused = self.branch_drop(fused)              # ← 进一步随机深度

        # 残差
        y = x_conv + self.alpha * fused

        # 时间维度平均池化 -> [B, outDims] -> 输出控制
        y = torch.mean(y, dim=1)
        return self.out_ctrl(y)


# =========================
# Delta & Hybrid 模块
# =========================

class Delta(nn.Module):
    def __init__(self, inDims, seqL, outDims=None, lstm_layers=1, use_residual=True,
                 use_mamba=True, mamba_impl='full', dropout=0.15,
                 drop_path_rate=0.10, out_init_temp=0.30, out_l2norm=True):
        super(Delta, self).__init__()
        self.inDims = inDims
        self.seqL = seqL
        self.use_residual = use_residual
        self.use_mamba = use_mamba

        weight = (np.ones(seqL, dtype=np.float32)) / (seqL / 2.0)
        weight[:seqL // 2] *= -1
        self.weight = nn.Parameter(torch.from_numpy(weight), requires_grad=False)

        self.outDims = outDims if outDims else inDims

        self.lstm = nn.LSTM(
            input_size=inDims,
            hidden_size=self.outDims,
            num_layers=lstm_layers,
            batch_first=True
        )

        if self.use_mamba:
            self.mamba = MambaWrapper(
                d_model=inDims,
                d_state=16,
                d_conv=4,
                expand=2,
                implementation=mamba_impl,
                num_layers=1,
                dropout=dropout,
                drop_path_rate=drop_path_rate
            )
            self.mamba_proj = nn.Linear(inDims, self.outDims) if self.outDims != inDims else None

        if self.use_residual and self.outDims != inDims:
            self.proj = nn.Linear(inDims, self.outDims)
        else:
            self.proj = None

        self.out_ctrl = OutController(self.outDims, init_temp=out_init_temp, normalize=out_l2norm)

    def forward(self, x):
        x_btc = x.permute(0, 2, 1)              # [B, C, T]
        delta = torch.matmul(x_btc, self.weight) # [B, C]
        delta_exp = delta.unsqueeze(1)           # [B, 1, C]

        if self.use_mamba:
            m_out = self.mamba(delta_exp)        # [B, 1, C]
            if self.mamba_proj:
                m_out = self.mamba_proj(m_out)   # [B, 1, outDims]
        else:
            m_out, _ = self.lstm(delta_exp)      # [B, 1, outDims]

        if self.use_residual:
            residual = delta                      # [B, C]
            if self.proj:
                residual = self.proj(residual)    # [B, outDims]
            out = m_out.squeeze(1) + residual     # [B, outDims]
        else:
            out = m_out.squeeze(1)                # [B, outDims]

        return self.out_ctrl(out)


class HybridSeqModel(nn.Module):
    """混合模型：结合 adapnet 和 Delta"""

    def __init__(self, inDims, seqL, hiddenDims=128, outDims=64, mamba_impl='full',
                 dropout=0.15, drop_path_rate=0.10, out_init_temp=0.30, out_l2norm=True):
        super(HybridSeqModel, self).__init__()

        self.seq_net = seqNet(inDims, hiddenDims, seqL, use_mamba=True, mamba_impl=mamba_impl,
                              dropout=dropout, drop_path_rate=drop_path_rate,
                              out_init_temp=out_init_temp, out_l2norm=out_l2norm)
        self.delta = Delta(inDims, seqL, hiddenDims, use_mamba=True, mamba_impl=mamba_impl,
                           dropout=dropout, drop_path_rate=drop_path_rate,
                           out_init_temp=out_init_temp, out_l2norm=out_l2norm)

        self.fusion = nn.Linear(hiddenDims * 2, outDims)

        self.final_mamba = MambaWrapper(
            d_model=outDims,
            d_state=16,
            d_conv=4,
            expand=2,
            implementation=mamba_impl,
            num_layers=1,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )

        # 最终输出也走一次温度+归一化，进一步钳制距离
        self.out_ctrl = OutController(outDims, init_temp=out_init_temp, normalize=out_l2norm)

    def forward(self, x):
        seq_feat = self.seq_net(x)                # [B, hiddenDims]
        delta_feat = self.delta(x)                # [B, hiddenDims]

        combined = torch.cat([seq_feat, delta_feat], dim=1)  # [B, hiddenDims*2]
        fused = self.fusion(combined)                         # [B, outDims]

        fused_exp = fused.unsqueeze(1)                        # [B, 1, outDims]
        final_out = self.final_mamba(fused_exp)               # [B, 1, outDims]
        final_out = final_out.squeeze(1)                      # [B, outDims]
        return self.out_ctrl(final_out)
