import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.pvtv2 import pvt_v2_b2


class BasicConv2d(nn.Module):
    def __init__(self, in_cha, out_cha, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_cha),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class cross_layer_attention(nn.Module):
    """
    原 CAE 里用的 cross-layer attention，保持不动
    """
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.num_heads = num_heads

        self.high_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.high_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.high_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.mask_q = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.norm_layer = nn.LayerNorm(dim)

    def forward(self, high_fea, mask):
        B, N, C = high_fea.shape

        high_q = self.high_q(high_fea).reshape(
            B, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        high_k = self.high_k(high_fea).reshape(
            B, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        high_v = self.high_v(high_fea).reshape(
            B, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)

        if mask is None:
            high_attn = torch.matmul(high_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = torch.matmul(high_attn, high_v).transpose(2, 1).reshape(B, N, C)
        else:
            mask_q = self.mask_q(mask).reshape(
                B, N, self.num_heads, C // self.num_heads
            ).permute(0, 2, 1, 3)

            high_attn = torch.matmul(mask_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = torch.matmul(high_attn, high_v).transpose(2, 1).reshape(B, N, C)

        return high_attn


class encoder_block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = cross_layer_attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class cross_attention_encoder(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio):
        super(cross_attention_encoder, self).__init__()
        self.depth = depth
        self.block = encoder_block(dim, num_heads, mlp_ratio)

    def forward(self, x, mask):
        for _ in range(self.depth):
            x = self.block(x, mask)
        return x


class SimpleTransformerBlock(nn.Module):
    """
    一个简化版 Transformer block，用于 decoder：
    标准 LayerNorm + MultiheadAttention + MLP 结构
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., drop_path=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SimpleTransformerDecoder(nn.Module):
    """
    由若干 SimpleTransformerBlock 堆叠而成的解码器
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, depth=4, drop=0., drop_path=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class TokenCrossAttention(nn.Module):
    """
    双向 Token 级 cross-attention：
    - x_tokens 做 Query，ref_tokens 做 Key/Value
    - ref_tokens 做 Query，x_tokens 做 Key/Value
    """
    def __init__(self, embed_dim, num_heads=8, qkv_bias=True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj_x = nn.LazyLinear(embed_dim, bias=qkv_bias)
        self.k_proj_ref = nn.LazyLinear(embed_dim, bias=qkv_bias)
        self.v_proj_ref = nn.LazyLinear(embed_dim, bias=qkv_bias)

        self.q_proj_ref = nn.LazyLinear(embed_dim, bias=qkv_bias)
        self.k_proj_x = nn.LazyLinear(embed_dim, bias=qkv_bias)
        self.v_proj_x = nn.LazyLinear(embed_dim, bias=qkv_bias)

        self.proj = nn.LazyLinear(embed_dim, bias=qkv_bias)

        self.norm_xq = nn.LayerNorm(embed_dim)
        self.norm_refq = nn.LayerNorm(embed_dim)

    def _attend(self, q, k, v):
        B, N, D = q.shape

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 1).reshape(B, N, D)

        return out

    def forward(self, x_tokens, ref_tokens):
        B, Nq, _ = x_tokens.shape
        Br, Nk, _ = ref_tokens.shape

        assert B == Br, "batch size 不一致"
        assert Nq == Nk, "token 数（空间尺寸）必须一致"

        q_x = self.q_proj_x(x_tokens)
        k_ref = self.k_proj_ref(ref_tokens)
        v_ref = self.v_proj_ref(ref_tokens)

        out_xq = self._attend(q_x, k_ref, v_ref)
        out_xq = self.proj(out_xq)
        out_xq = self.norm_xq(out_xq)

        q_ref = self.q_proj_ref(ref_tokens)
        k_x = self.k_proj_x(x_tokens)
        v_x = self.v_proj_x(x_tokens)

        out_refq = self._attend(q_ref, k_x, v_x)
        out_refq = self.proj(out_refq)
        out_refq = self.norm_refq(out_refq)

        return out_xq, out_refq


class SoftGateCrossAttention(nn.Module):
    """
    BTCAF：
    双向 cross-att + 两个独立 gate。

    out = x + g1 * out_xq + g2 * out_refq
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        self.cross_att = TokenCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        self.gate_mlp1 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.gate_mlp2 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x_tokens, ref_tokens):
        out_xq, out_refq = self.cross_att(x_tokens, ref_tokens)

        gate_input = torch.cat([x_tokens, ref_tokens], dim=-1)

        g1 = self.gate_mlp1(gate_input)
        g2 = self.gate_mlp2(gate_input)

        out = x_tokens + g1 * out_xq + g2 * out_refq

        return out


class PolarSectorSelfAttn(nn.Module):
    """
    RAFA 稳定版:
    极坐标扇区 token + self-attention 的频域增强模块。

    关键修改：
    1. 不再使用 torch.angle(S) + exp(1j * P)
    2. 使用 phase = S / (abs(S) + eps)，并 detach phase
    3. A_base = abs(S).detach()
    4. FFT 统计分支不向输入特征反传梯度
    5. RAFA token encoder 固定 fp32，减少 AMP 下半精度不稳定

    输出仍然是：
        out = F_in + tanh(gamma) * att * F_in

    在 Network 里继续弱注入：
        x = x + beta * (RAFA(x) - x)
    """
    def __init__(
        self,
        Nr=8,
        Ntheta=16,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        eps=1e-6,
        m_clamp=6.0,
        gamma_init=0.1
    ):
        super().__init__()

        self.Nr = Nr
        self.Ntheta = Ntheta
        self.Nsec = Nr * Ntheta
        self.eps = eps
        self.m_clamp = m_clamp

        self.in_proj = nn.Linear(1, embed_dim)

        self.r_emb = nn.Embedding(Nr, embed_dim)
        self.t_emb = nn.Embedding(Ntheta, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation='gelu',
            dropout=0.0
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=depth
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self._cache = {}

    @torch.no_grad()
    def _build_sector_maps(self, H, W, device):
        key = (H, W, device)

        if key in self._cache:
            return self._cache[key]

        ys = torch.arange(H, device=device) - (H // 2)
        xs = torch.arange(W, device=device) - (W // 2)

        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        r_max = r.max().clamp(min=1.0)
        r_norm = r / r_max

        r_id = torch.clamp(
            (r_norm * self.Nr).long(),
            0,
            self.Nr - 1
        )

        theta = torch.atan2(yy.float(), xx.float())
        theta_norm = (theta + math.pi) / (2 * math.pi)

        t_id = torch.clamp(
            (theta_norm * self.Ntheta).long(),
            0,
            self.Ntheta - 1
        )

        sector_id = (r_id * self.Ntheta + t_id).view(-1)

        sec_ids = torch.arange(self.Nsec, device=device)

        # 避免 PyTorch 关于 // 的 warning
        r_tok = torch.div(sec_ids, self.Ntheta, rounding_mode='floor').long()
        t_tok = (sec_ids % self.Ntheta).long()

        self._cache[key] = (sector_id, r_tok, t_tok)

        return self._cache[key]

    def forward(self, F_in):
        B, C, H, W = F_in.shape
        device = F_in.device
        orig_dtype = F_in.dtype

        # [B, C, H, W] -> [B, 1, H, W]
        G = torch.mean(F_in, dim=1, keepdim=True)

        # =====================================================
        # FFT 统计分支
        # -----------------------------------------------------
        # 这里只把频谱当作条件输入，不让梯度穿过输入频谱。
        # 这样可以避免 abs / phase / ifft 对主路径产生不稳定梯度。
        # =====================================================
        with torch.cuda.amp.autocast(enabled=False):
            G32 = G.float()

            S = torch.fft.fft2(G32, dim=(-2, -1))
            S = torch.fft.fftshift(S, dim=(-2, -1))

            A = torch.abs(S)

            # 关键稳定处理：
            # A_base 用于扇区统计和幅值重建，但不对输入反传。
            A_base = A.detach()

            # 用单位复数相位替代 torch.angle + exp(1jP)
            # phase 同样 detach，避免低幅值位置相位梯度不稳定。
            phase = (S / (A + self.eps)).detach()

            sector_id, r_tok, t_tok = self._build_sector_maps(H, W, device)

            sector_id_b = sector_id.view(1, 1, -1).expand(B, 1, -1)
            A_flat = A_base.view(B, 1, -1)

            sums = torch.zeros(
                B, 1, self.Nsec,
                device=device,
                dtype=A_base.dtype
            )

            cnts = torch.zeros(
                B, 1, self.Nsec,
                device=device,
                dtype=A_base.dtype
            )

            ones = torch.ones_like(A_flat)

            sums.scatter_add_(dim=2, index=sector_id_b, src=A_flat)
            cnts.scatter_add_(dim=2, index=sector_id_b, src=ones)

            stats = sums / (cnts + self.eps)
            tokens = stats.transpose(1, 2).contiguous()

        # =====================================================
        # RAFA token encoder
        # -----------------------------------------------------
        # 固定 fp32，避免 AMP 下 TransformerEncoder 半精度不稳定。
        # =====================================================
        with torch.cuda.amp.autocast(enabled=False):
            tokens = tokens.float()

            x = self.in_proj(tokens)

            r_pos = self.r_emb(r_tok)[None, :, :].float()
            t_pos = self.t_emb(t_tok)[None, :, :].float()

            x = x + r_pos + t_pos
            x = self.encoder(x)

            w_sec = self.out_proj(x).squeeze(-1)

            w_sec_ = w_sec[:, None, :]
            w_flat = torch.gather(
                w_sec_,
                dim=2,
                index=sector_id_b
            )

            Wmap = w_flat.view(B, 1, H, W).float()

            # 频谱幅值调制
            A2 = A_base * (0.5 + 0.5 * Wmap)

            # 不再使用 angle，直接用 detach 后的单位复数相位
            S2 = A2 * phase
            S2 = torch.fft.ifftshift(S2, dim=(-2, -1))

            M = torch.fft.ifft2(S2, dim=(-2, -1)).real

            m_mean = M.mean(dim=(-2, -1), keepdim=True)
            m_std = M.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

            M = (M - m_mean) / m_std
            M = torch.clamp(M, -self.m_clamp, self.m_clamp)

            att = torch.sigmoid(M)

        att = att.to(dtype=orig_dtype)

        scale = torch.tanh(self.gamma).to(dtype=orig_dtype)

        out = F_in + scale * att * F_in

        return out
class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.imgsize = opt.imgsize

        self.backbone = pvt_v2_b2()

        path = opt.pvt_weights if hasattr(opt, 'pvt_weights') else './pvt_weights/pvt_v2_b2.pth'

        save_model = torch.load(path, map_location='cpu')
        model_dict = self.backbone.state_dict()

        state_dict = {
            k: v
            for k, v in save_model.items()
            if k in model_dict.keys()
        }

        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.dim_out = opt.dim
        self.use_polar = bool(getattr(opt, 'use_polar', 1))

        self.sigmoid = nn.Sigmoid()

        self.soft_split = nn.Unfold(
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0)
        )

        self.soft_fuse = nn.Fold(
            output_size=(self.imgsize // 4, self.imgsize // 4),
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0)
        )

        self.ref_conv0 = BasicConv2d(64, self.dim_out, kernel_size=1, stride=1, padding=0)
        self.ref_conv1 = BasicConv2d(64, self.dim_out, kernel_size=1, stride=1, padding=0)
        self.ref_conv2 = BasicConv2d(64, self.dim_out, kernel_size=1, stride=1, padding=0)
        self.ref_conv3 = BasicConv2d(64, self.dim_out, kernel_size=1, stride=1, padding=0)

        self.conv0 = BasicConv2d(64, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv1 = BasicConv2d(128, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(320, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(512, self.dim_out, kernel_size=3, stride=1, padding=1)

        self.conv_out = nn.Conv2d(
            self.dim_out,
            1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        dim_token = self.dim_out * 4 * 4

        self.fusion3 = SoftGateCrossAttention(embed_dim=dim_token, num_heads=8)
        self.fusion2 = SoftGateCrossAttention(embed_dim=dim_token, num_heads=8)
        self.fusion1 = SoftGateCrossAttention(embed_dim=dim_token, num_heads=8)
        self.fusion0 = SoftGateCrossAttention(embed_dim=dim_token, num_heads=8)

        self.TransformerEncoder3 = cross_attention_encoder(
            dim=dim_token,
            num_heads=16,
            depth=2,
            mlp_ratio=4.
        )

        self.TransformerEncoder2 = cross_attention_encoder(
            dim=dim_token,
            num_heads=16,
            depth=2,
            mlp_ratio=4.
        )

        self.TransformerEncoder1 = cross_attention_encoder(
            dim=dim_token,
            num_heads=16,
            depth=3,
            mlp_ratio=4.
        )

        self.TransformerEncoder0 = cross_attention_encoder(
            dim=dim_token,
            num_heads=16,
            depth=4,
            mlp_ratio=4.
        )

        self.TransDecoder = SimpleTransformerDecoder(
            dim=dim_token,
            num_heads=8,
            mlp_ratio=4.0,
            depth=4
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        # =====================================================
        # All-stage RAFA
        # -----------------------------------------------------
        # stage0 / stage1 / stage2 / stage3 各自使用独立 RAFA。
        # 不共享模块，避免不同语义层级的频域调制互相干扰。
        #
        # 默认 rafa_stages='0123'，也就是 all stage。
        # 如果之后想做消融，可以传：
        #   '3'    -> 只跑 stage3
        #   '23'   -> 跑 stage2 + stage3
        #   '123'  -> 跑 stage1 + stage2 + stage3
        #   '0123' -> 跑 all stage
        # =====================================================
        self.polar = nn.ModuleList([
            PolarSectorSelfAttn(
                Nr=8,
                Ntheta=16,
                embed_dim=64,
                depth=2,
                num_heads=4,
                gamma_init=getattr(opt, 'rafa_gamma_init', 0.1)
            )
            for _ in range(4)
        ])

        self.rafa_stages = str(getattr(opt, 'rafa_stages', '0123'))

        # =====================================================
        # BTCAF 后 RAFA 弱注入系数
        # -----------------------------------------------------
        # 因为你已经观察到：
        #   1) RAFA 直接加 stage3 会降指标；
        #   2) RAFA 加在 BTCAF 后也可能降指标；
        #
        # 所以这里把 beta 设置得非常保守。
        #
        # 默认：
        #   rafa_max_beta  = 0.03   最大只允许 3% 的 RAFA 残差注入
        #   rafa_beta_init = 0.003  初始只注入约 0.3%
        #
        # 融合公式：
        #   x = x + beta * (RAFA(x) - x)
        # =====================================================
        self.rafa_max_beta = float(getattr(opt, 'rafa_max_beta', 0.03))
        rafa_beta_init = float(getattr(opt, 'rafa_beta_init', 0.003))

        ratio = rafa_beta_init / max(self.rafa_max_beta, 1e-6)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)

        beta_logit = math.log(ratio / (1.0 - ratio))

        # 每个 stage 一个独立 beta。
        # all-stage 时不要共用同一个 beta，否则底层和高层会被同一强度强行绑定。
        self.rafa_beta_logit = nn.Parameter(
            torch.full((4,), beta_logit, dtype=torch.float32)
        )

        self.upsample2 = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        )

        self.upsample4 = nn.Upsample(
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        )

        self.upsample8 = nn.Upsample(
            scale_factor=8,
            mode='bilinear',
            align_corners=True
        )

    def _ensure_4d(self, t, x_like):
        """
        把 ref_feat 调整到 [B,C,H,W]，并对齐 dtype/device。
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError("ref_feats must be torch.Tensor")

        if t.dim() == 3:
            t = t.unsqueeze(0)

        return t.to(dtype=x_like.dtype, device=x_like.device)

    def _post_btcaf_rafa_injection(self, tokens, stage_idx):
        """
        在指定 stage 的 BTCAF 后进行 RAFA 弱注入。

        输入：
            tokens: [B, N, dim_out * 16]
            stage_idx: 0 / 1 / 2 / 3

        流程：
            token -> feature map -> RAFA -> weak residual -> token

        融合公式：
            x = x + beta_s * (RAFA_s(x) - x)

        输出：
            tokens: [B, N, dim_out * 16]
        """
        fused_map = self.soft_fuse(tokens.transpose(-2, -1))

        rafa_map = self.polar[stage_idx](fused_map)

        beta = self.rafa_max_beta * torch.sigmoid(
            self.rafa_beta_logit[stage_idx]
        )

        fused_map = fused_map + beta * (rafa_map - fused_map)

        tokens = self.soft_split(fused_map).transpose(-2, -1)

        return tokens

    def forward(self, x, ref_feats, y=None, training=True):
        """
        ref_feats: (r1, r2, r3, r4)

        training=True:
            return s3_out, s2_out, s1_out, s0_out, loss_prob, err_map, u

        training=False:
            return s3_out, s2_out, s1_out, s0_out, None, None, None
        """
        B, _, _, _ = x.shape

        pvt = self.backbone(x)

        x0, x1, x2, x3 = pvt[0], pvt[1], pvt[2], pvt[3]

        r1, r2, r3, r4 = ref_feats

        r1 = self._ensure_4d(r1, x)
        r2 = self._ensure_4d(r2, x)
        r3 = self._ensure_4d(r3, x)
        r4 = self._ensure_4d(r4, x)

        x0_ = self.conv0(x0)
        x1_ = self.conv1(x1)
        x2_ = self.conv2(x2)
        x3_ = self.conv3(x3)

        ref0 = self.ref_conv0(r1)
        ref1 = self.ref_conv1(r2)
        ref2 = self.ref_conv2(r3)
        ref3 = self.ref_conv3(r4)

        x3_up = self.upsample8(x3_)
        ref3_up = F.interpolate(
            ref3,
            size=x3_up.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x2_up = self.upsample4(x2_)
        ref2_up = F.interpolate(
            ref2,
            size=x2_up.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x1_up = self.upsample2(x1_)
        ref1_up = F.interpolate(
            ref1,
            size=x1_up.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        x0_up = x0_
        ref0_up = F.interpolate(
            ref0,
            size=x0_up.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # =====================================================
        # Stage3:
        # 先 BTCAF，后 RAFA 弱注入
        # =====================================================
        x3_tokens = self.soft_split(x3_up).transpose(-2, -1)
        r3_tokens = self.soft_split(ref3_up).transpose(-2, -1)

        x3_tokens = self.fusion3(x3_tokens, r3_tokens)

        if self.use_polar and ('3' in self.rafa_stages):
            x3_tokens = self._post_btcaf_rafa_injection(
                x3_tokens,
                stage_idx=3
            )

        # =====================================================
        # Stage2:
        # 先 BTCAF，后 RAFA 弱注入
        # =====================================================
        x2_tokens = self.soft_split(x2_up).transpose(-2, -1)
        r2_tokens = self.soft_split(ref2_up).transpose(-2, -1)

        x2_tokens = self.fusion2(x2_tokens, r2_tokens)

        if self.use_polar and ('2' in self.rafa_stages):
            x2_tokens = self._post_btcaf_rafa_injection(
                x2_tokens,
                stage_idx=2
            )

        # =====================================================
        # Stage1:
        # 先 BTCAF，后 RAFA 弱注入
        # =====================================================
        x1_tokens = self.soft_split(x1_up).transpose(-2, -1)
        r1_tokens = self.soft_split(ref1_up).transpose(-2, -1)

        x1_tokens = self.fusion1(x1_tokens, r1_tokens)

        if self.use_polar and ('1' in self.rafa_stages):
            x1_tokens = self._post_btcaf_rafa_injection(
                x1_tokens,
                stage_idx=1
            )

        # =====================================================
        # Stage0:
        # 先 BTCAF，后 RAFA 弱注入
        # =====================================================
        x0_tokens = self.soft_split(x0_up).transpose(-2, -1)
        r0_tokens = self.soft_split(ref0_up).transpose(-2, -1)

        x0_tokens = self.fusion0(x0_tokens, r0_tokens)

        if self.use_polar and ('0' in self.rafa_stages):
            x0_tokens = self._post_btcaf_rafa_injection(
                x0_tokens,
                stage_idx=0
            )

        s3 = self.TransformerEncoder3(x3_tokens, mask=None)
        s2 = self.TransformerEncoder2(x2_tokens, s3)
        s1 = self.TransformerEncoder1(x1_tokens, s2)
        s0 = self.TransformerEncoder0(x0_tokens, s1)

        s3 = self.TransDecoder(s3)
        s2 = self.TransDecoder(s2 + s3)
        s1 = self.TransDecoder(s1 + s2)
        s0 = self.TransDecoder(s0 + s1)

        s3_map = self.soft_fuse(s3.transpose(-2, -1))
        s2_map = self.soft_fuse(s2.transpose(-2, -1))
        s1_map = self.soft_fuse(s1.transpose(-2, -1))
        s0_map = self.soft_fuse(s0.transpose(-2, -1))

        s3_out = self.upsample4(self.conv_out(s3_map))
        s2_out = self.upsample4(self.conv_out(s2_map))
        s1_out = self.upsample4(self.conv_out(s1_map))
        s0_out = self.upsample4(self.conv_out(s0_map))

        s3_out = torch.nan_to_num(torch.clamp(s3_out, -20.0, 20.0))
        s2_out = torch.nan_to_num(torch.clamp(s2_out, -20.0, 20.0))
        s1_out = torch.nan_to_num(torch.clamp(s1_out, -20.0, 20.0))
        s0_out = torch.nan_to_num(torch.clamp(s0_out, -20.0, 20.0))

        loss_prob = torch.zeros(1, device=x.device, dtype=x.dtype)
        err_map = None
        u = None

        if training and (y is not None):
            y_resized = F.interpolate(
                y,
                size=s0_out.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            y_resized = torch.clamp(y_resized, 0.0, 1.0)

            with torch.no_grad():
                pred_prob = torch.sigmoid(s0_out)
                err_map = torch.abs(pred_prob - y_resized)

            u_logits = self.uncertainty_head(s0_out)
            u = torch.sigmoid(u_logits)

            loss_prob = F.l1_loss(
                u,
                err_map,
                reduction='mean'
            )

        if training:
            return s3_out, s2_out, s1_out, s0_out, loss_prob, err_map, u
        else:
            return s3_out, s2_out, s1_out, s0_out, None, None, None