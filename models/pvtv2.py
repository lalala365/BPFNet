import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

import math

class Mlp(nn.Module):
    """多层感知机模块，在标准MLP基础上加入了深度可分离卷积"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一个全连接层：扩展特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 深度可分离卷积：增强局部特征提取能力（PVTv2的关键创新）
        self.dwconv = DWConv(hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二个全连接层：恢复特征维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层防止过拟合
        self.drop = nn.Dropout(drop)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            # 线性层使用截断正态分布初始化
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm层初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            # 卷积层使用Kaiming初始化
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """前向传播
        Args:
            x: 输入张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        Returns:
            x: 输出张量 [B, N, C]
        """
        # 第一个全连接层
        x = self.fc1(x)
        # 深度可分离卷积（需要空间维度信息）
        x = self.dwconv(x, H, W)
        # 激活函数
        x = self.act(x)
        # Dropout
        x = self.drop(x)
        # 第二个全连接层
        x = self.fc2(x)
        # Dropout
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """自注意力机制模块，包含空间缩减注意力(SRA)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 注意力缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # Q、K、V投影层
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 空间缩减比率（SRA的关键参数）
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 空间缩减卷积：通过卷积降采样减少K、V的序列长度
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """前向传播
        Args:
            x: 输入张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        Returns:
            x: 注意力加权后的输出 [B, N, C]
        """
        B, N, C = x.shape
        
        # Q投影和reshape: [B, N, C] -> [B, num_heads, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # K、V投影（包含空间缩减）
        if self.sr_ratio > 1:
            # 空间缩减：通过卷积降低K、V的分辨率以减少计算量
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, N, C] -> [B, C, H, W]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # [B, C, H/sr, W/sr] -> [B, N/sr^2, C]
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            # 无空间缩减
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]  # 分离K和V

        # 注意力计算: Q * K^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # Softmax归一化
        attn = self.attn_drop(attn)  # 注意力Dropout

        # 注意力加权: attn * V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Transformer基础块，包含注意力层和MLP层"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        # 第一个归一化层
        self.norm1 = norm_layer(dim)
        # 注意力层
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # 随机深度衰减（DropPath）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个归一化层
        self.norm2 = norm_layer(dim)
        # MLP层
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """前向传播：残差连接 + 层归一化 + 注意力/MLP"""
        # 注意力子层：LayerNorm -> Attention -> DropPath -> 残差连接
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # MLP子层：LayerNorm -> MLP -> DropPath -> 残差连接
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """重叠的图像块嵌入层，将图像分割为重叠的patch并投影到特征空间"""
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # 计算patch数量
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # 重叠的卷积投影层：使用重叠的卷积核保留更多空间信息
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """前向传播：图像 -> 特征图 -> 序列化 -> 归一化"""
        # 卷积投影: [B, C, H, W] -> [B, embed_dim, H', W']
        x = self.proj(x)
        _, _, H, W = x.shape
        # 展平为序列: [B, embed_dim, H', W'] -> [B, embed_dim, H'W'] -> [B, H'W', embed_dim]
        x = x.flatten(2).transpose(1, 2)
        # 层归一化
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    """改进的金字塔视觉Transformer主干网络（PVTv2）"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # 四个阶段的patch embedding层（渐进式下采样）
        # 阶段1: 1/4下采样
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        # 阶段2: 1/8下采样  
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        # 阶段3: 1/16下采样
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        # 阶段4: 1/32下采样
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # 随机深度衰减规则（Stochastic Depth）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # 四个阶段的Transformer块
        # 阶段1: 高分辨率，浅层特征
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        # 阶段2: 中等分辨率
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        # 阶段3: 较低分辨率
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        # 阶段4: 低分辨率，深层语义特征
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # 分类头（在UAT中不使用）
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """预训练权重加载"""
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        """重置DropPath概率"""
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        """冻结patch embedding层"""
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """不进行权重衰减的参数"""
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        """获取分类器"""
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """重置分类器"""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """特征提取前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            outs: 多尺度特征图列表 [stage1, stage2, stage3, stage4]
        """
        B = x.shape[0]
        outs = []  # 存储多尺度特征图
        outs_tokens = []  # 存储token序列（可选）

        # ========== 阶段1: 1/4分辨率 ==========
        # patch embedding: [B, 3, 352, 352] -> [B, L1, C1=64], H1=88, W1=88
        x, H, W = self.patch_embed1(x)
        # Transformer块处理
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        # 层归一化
        x = self.norm1(x)
        outs_tokens.append(x)
        # 重塑为特征图格式: [B, L1, C1] -> [B, C1, H1, W1]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)  # [B, 64, 88, 88]

        # ========== 阶段2: 1/8分辨率 ==========
        x, H, W = self.patch_embed2(x)  # [B, 64, 88, 88] -> [B, L2, C2=128], H2=44, W2=44
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        outs_tokens.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)  # [B, 128, 44, 44]

        # ========== 阶段3: 1/16分辨率 ==========
        x, H, W = self.patch_embed3(x)  # [B, 128, 44, 44] -> [B, L3, C3=320], H3=22, W3=22
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        outs_tokens.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)  # [B, 320, 22, 22]

        # ========== 阶段4: 1/32分辨率 ==========
        x, H, W = self.patch_embed4(x)  # [B, 320, 22, 22] -> [B, L4, C4=512], H4=11, W4=11
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        outs_tokens.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)  # [B, 512, 11, 11]

        return outs  # 返回多尺度特征图
        # return outs_tokens  # 或者返回token序列

    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            x: 多尺度特征图列表
        """
        x = self.forward_features(x)
        # x = self.head(x)  # 分类头（在UAT中不使用）
        return x


class DWConv(nn.Module):
    """深度可分离卷积，用于MLP中的局部特征增强"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        # 深度可分离卷积：每个通道独立卷积
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """前向传播
        Args:
            x: 输入张量 [B, N, C]
            H: 特征图高度
            W: 特征图宽度
        Returns:
            x: 卷积后的输出 [B, N, C]
        """
        B, N, C = x.shape
        # 重塑为2D格式: [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        # 深度可分离卷积
        x = self.dwconv(x)
        # 恢复为序列格式: [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        return x


def _conv_filter(state_dict, patch_size=16):
    """转换patch embedding权重：从手动patchify + 线性投影转换为卷积"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# ========== PVTv2不同规模的模型配置 ==========

@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    """PVTv2-B0: 最小模型，计算量最小"""
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, 
            embed_dims=[32, 64, 160, 256],  # 四个阶段的通道数
            num_heads=[1, 2, 5, 8],         # 四个阶段的注意力头数
            mlp_ratios=[8, 8, 4, 4],        # MLP扩展比率
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2],            # 四个阶段的块数
            sr_ratios=[8, 4, 2, 1],         # 四个阶段的空间缩减比率
            drop_rate=0.0, 
            drop_path_rate=0.1)


@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    """PVTv2-B1: 小型模型"""
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[2, 2, 2, 2], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)


@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    """PVTv2-B2: 中型模型，论文中常用的配置"""
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 6, 3],            # 更深的网络结构
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)


@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    """PVTv2-B3: 大型模型"""
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 18, 3],           # 第三阶段特别深
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)


@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    """PVTv2-B4: 超大型模型"""
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 8, 27, 3],           # 更深的第二、三阶段
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)


@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    """PVTv2-B5: 最大模型"""
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8], 
            mlp_ratios=[4, 4, 4, 4],        # 较小的MLP比率
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 6, 40, 3],           # 非常深的第三阶段
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1)