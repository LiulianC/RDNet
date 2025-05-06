# --------------------------------------------------------
# FocalNet for Semantic Segmentation
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang
# --------------------------------------------------------
import math
import time
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Mlp 类是一个 ​​多层感知机（Multilayer Perceptron, MLP）模块​​，
# 广泛用于深度学习模型中实现特征的​​非线性变换和增强​​
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) # Dropout 比率（默认 0，即无丢弃）

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x






# FocalModulation 是一种 ​​多尺度动态特征增强模块​​，
# 通过层级化的卷积操作和门控机制，高效融合局部与全局上下文。
# 其名称体现了 ​​多焦点（多尺度）​​ 和 ​​动态调制​​ 的核心设计理念，
# 适用于需要平衡计算效率与模型性能的视觉任务。
class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels 
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False, 
        use_postln_in_modulation=False, normalize_modulator=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level # 焦点层级数（如2层表示使用两种不同尺度的卷积核）
        self.focal_window = focal_window # 基础卷积核大小（如7x7）
        self.focal_factor = focal_factor # 	每层卷积核扩展步长（第k层核大小：focal_window + focal_factor*k）
        self.use_postln_in_modulation = use_postln_in_modulation # 	是否在输出前应用层归一化
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True) # 生成查询 q、初始上下文 ctx 和各层门控 gates
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True) # 	将融合后的上下文映射到调制空间

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList() # 	多尺度卷积层（不同核大小的深度可分离卷积）

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        # 这是多尺度卷积
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, H, W, C)
        """
        B, nH, nW, C = x.shape
        x = self.f(x) # 通过线性层生成多通道 生成查询 q、初始上下文 ctx 和各层门控 gates的通道
        x = x.permute(0, 3, 1, 2).contiguous() # (N,H,W,C) -> (N, C, H, W).contiguous()​​ 确保张量在内存中连续存储
        
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        # 输入 x 的形状​​：假设为 (B, C_total, H, W)，其中 C_total = C + C + (self.focal_level+1)。
        # q：形状 (B, C, H, W)，表示查询（Query）特征。通过线性变换从输入中提取 ​​与位置强相关​​ 的特征
        # ctx：形状 (B, C, H, W)，表示上下文（Context）特征。通过后续的 ​​多尺度卷积​​ 和 ​​门控融合​​ 提取上下文
        # gates：形状 (B, self.focal_level+1, H, W)，表示多焦点级别的门控权重。动态调节不同尺度特征的贡献权重
        # ​尽管 q、ctx、gates 来自x经过同一线性层映射出来的三个区域，但模型可以通过反向传播自动学习如何将输入特征分解为三个功能不同的部分。
        
        # ​​多尺度上下文提取​​：对 ctx 依次进行不同尺度的卷积操作：
        ctx_all = 0
        for l in range(self.focal_level):  # # 焦点层级数（如2层表示使用两种不同尺度的卷积核）            
            ctx = self.focal_layers[l](ctx) # 送入多尺度卷积 通道数不变
            # gates[:, l:l+1] 选的是第l时候的gates，是形状为 [B, 1, H, W] 的权重张量，用于空间自适应的加权。
            # 若使用 gates[:, l]（形状 [B, H, W]），其维度与 ctx ​​无法直接对齐
            ctx_all = ctx_all + ctx*gates[:, l:l+1]  

        # ​全局上下文补充​​：通过全局平均池化提取全局上下文，并与门控权重融合：
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True)) # [B, C, H, W] → [B, C, 1, W] -> [B, C, 1, 1] 全局平均池化
        # gates最后一个通道的维度（形状 [B, 1, H, W]）
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1) # 除以这么多次是因为之前加了这么多次

        #  # 特征调制​​：将融合后的上下文 ctx_all 与查询 q 相乘，通过 self.h（1*1卷积） 调整q与ctx通道维度相等：
        # 若 modulation 某位置通道值 > 1 → 增强 q 对应位置的响应（重要特征）。
        x_out = q * self.h(ctx_all)

        # ​​输出归一化与投影​​：
        x_out = x_out.permute(0, 2, 3, 1).contiguous() #（B,C,H,W）->（B,H,W,C）
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)   # 归一化     
        x_out = self.proj(x_out)    # 线性层调整通道数
        x_out = self.proj_drop(x_out) # 随机丢弃 在前向传播时，对投影层（self.proj）的输出以概率 proj_drop 随机置零部分神经元，缓解过拟合
        return x_out



# 一种 ​​高效的特征增强模块​​，替代传统自注意力（Self-Attention）或卷积，用于视觉任务中的多尺度特征建模。
# "Focal"​​：表示该模块采用 ​​多尺度/多焦点​​ 的上下文聚合机制，类似人类视觉的焦点切换，捕捉不同范围的局部-全局信息。
# ​​"Modulation"​​：指 ​​动态特征调整​​，通过门控机制自适应融合多尺度特征。
# ​​"Block"​​：表明这是一个 ​​基础网络模块​​，可堆叠构建深度模型（类似 Transformer Block）。
class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.

    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, 
                 use_postln=False, use_postln_in_modulation=False, 
                 normalize_modulator=False, 
                 use_layerscale=False, 
                 layerscale_value=1e-4):
        super().__init__()

        # # 参数保存
        self.dim = dim
        self.mlp_ratio = mlp_ratio # MLP隐藏层维度是输入维度的mlp_ratio倍
        self.focal_window = focal_window
        self.focal_level = focal_level

        self.use_postln = use_postln
        self.use_layerscale = use_layerscale

        # 核心模块
        self.norm1 = norm_layer(dim)# 前置归一化
        self.modulation = FocalModulation( # 多尺度调制模块
            dim, focal_window=self.focal_window, focal_level=self.focal_level, proj_drop=drop, 
            use_postln_in_modulation=use_postln_in_modulation, 
            normalize_modulator=normalize_modulator, 
        )            
        self.norm2 = norm_layer(dim) # MLP前归一化
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 残差连接与正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # DropPath随机深度丢弃;nn.Identity()是输入什么就输出什么，不会改变数据的形状或数值。这有助于保持计算图的完整性，同时允许在某些情况下灵活地启用或禁用某些层
        self.gamma_1 = 1.0 # LayerScale可学习参数
        self.gamma_2 = 1.0 # LayerScale可学习参数

        self.H = None
        self.W = None
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape # 输入形状 [Batch, SeqLen, Channels]
        H, W = self.H, self.W  # 预设的空间分辨率（如14x14） 调用时通过blk.H和blk.W传入,H W是本模块的一个属性
        # assert 是 Python 中的一个关键字，用于调试时测试条件是否为真。如果条件为假，则会引发 AssertionError 异常
        assert L == H * W, "input feature has wrong size"  # 确保序列长度等于H*W
        shortcut = x # 保存残差连接的原始输入

        # 前置归一化（若未启用Post-LN）
        # use_postln=False（默认）​​: 使用 ​​前置归一化（Pre-LN）​​，即先对输入进行归一化，再进行调制（Modulation）或 MLP 操作。
        # ​use_postln=True​​: 使用 ​​后置归一化（Post-LN）​​，即先进行调制或 MLP 操作，再对结果进行归一化，并与残差连接相加。
        if not self.use_postln: 
            x = self.norm1(x) # LayerNorm处理 [B, L, C]
        x = x.view(B, H, W, C) # 将序列转换为2D特征图 # [B, H*W, C] → [B, H, W, C]
        
        # FM # 通过FocalModulation模块处理
        x = self.modulation(x).view(B, H * W, C) # 输入 [B, H, W, C] →  [B, H, W, C] -> [B, H*W, C]
        if self.use_postln: # 后置归一化（若启用Post-LN）
            x = self.norm1(x)  # 替代前置归一化

        # FFN # 第一次残差连接（调制后特征 + 原始输入）
        x = shortcut + self.drop_path(self.gamma_1 * x) # self.gamma_1初始为1或可学习的小值（LayerScale）

        # 归一化与MLP处理
        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x))) # 标准前置归一化# 标准前置归一化# 第二次残差连接
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# BasicLayer 类是一个 ​​多阶段视觉模型的基础层级结构​​，
# 用于整合特征变换模块（如 FocalModulationBlock）和空间下采样操作，逐步提取多尺度特征。
# ​​"Basic"​​：表明这是网络中的一个 ​​基础构建单元​​，通常作为主干网络的阶段模块。
# ​​"Layer"​​：指代模型的一个 ​​处理阶段​​（Stage），包含多个特征变换块（Block）和可选的层级间下采样。
class BasicLayer(nn.Module):
    """ A basic focal modulation layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Use overlapped convolution for patch embedding or now. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_window=9, 
                 focal_level=2, 
                 use_conv_embed=False,     
                 use_postln=False,          
                 use_postln_in_modulation=False, 
                 normalize_modulator=False, 
                 use_layerscale=False,                   
                 use_checkpoint=False
        ):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([ # 通过堆叠 depth 个 FocalModulationBlock 实现连续的 ​​局部-全局特征交互​​：
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                focal_window=focal_window, 
                focal_level=focal_level, 
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator, 
                use_layerscale=use_layerscale, 
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None: # 通过 downsample 模块（通常为步长2的卷积或补丁合并）实现 ​​分辨率减半、通道翻倍​​。
            self.downsample = downsample(
                patch_size=2, 
                in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C). 
            H, W: Spatial resolution of the input feature.
            输入张量​​：x 形状为 [B, H*W, C]（序列化特征，类似 Vision Transformer 的 Patch 嵌入）。
        """

        for blk in self.blocks: # 将 H, W 传递给每个 FocalModulationBlock，并逐块处理输入特征。
            blk.H, blk.W = H, W # 将当前的​​特征图尺寸​​（高度 H 和宽度 W）传递给每个块（如 FocalModulationBlock）
            if self.use_checkpoint: # 启用 ​​梯度检查点技术 训练时​​：前向传播中不保留中间激活值，反向传播时重新计算这些值。 ​测试时​​：无影响，等价于直接调用 blk(x)。
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x) # 若未启用检查点，直接执行块的前向传播。
                # 循环处理x这么多次 有什么用吗？
        
        # 可选下采样​​：
        if self.downsample is not None:
            x_reshaped = x.transpose(1, 2).view(x.shape[0], x.shape[-1], H, W) # 序列 → 2D 图像格式
            x_down = self.downsample(x_reshaped)      # 卷积降采样 + 通道扩展
            x_down = x_down.flatten(2).transpose(1, 2)            # 2D → 序列格式
            Wh, Ww = (H + 1) // 2, (W + 1) // 2      # 新分辨率
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W



# ​​类名​​：PatchEmbed
# ​​"Patch"​​：指将图像分割为若干 ​​局部块​​（如 4x4 像素的网格）。
# ​​"Embed"​​：表示将每个块 ​​映射为低维向量​​（嵌入向量）。
# ​​组合意义​​：​​图像块嵌入模块​​，核心功能是将图像转换为一系列可处理的嵌入向量序列，类似 Vision Transformer (ViT) 的预处理步骤。
# ​​定位​​：视觉模型的 ​​输入预处理模块​​，负责将原始像素转换为结构化特征表示。

# 分块是通过卷积后通道数增加 而HW面积减少来实现的
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96. 线性投影输出通道数（维度）
        norm_layer (nn.Module, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, use_conv_embed=False, is_stem=False):
        super().__init__() # 若use_conv_embed=True，则采用 ​​重叠卷积​​ 替代直接分块，增强局部特征连续性
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size # 根据 patch_size 或卷积步长（如步长 2）​​降低空间分辨率​​，同时 ​​扩展通道维度​​（如 3 通道 → 96 通道）。

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed: # True：使用 ​​卷积实现嵌入​​（类似 ResNet 的 Stem 层），通过 kernel_size 和 stride 控制分块方式
            if is_stem: # Stem 模式位于网络的最前端（即“茎干”位置，故称 Stem）快速降低输入图像的分辨率，​​ 保留局部信息​​，为后续的深层卷积或 Transformer 块提供低分辨率、高通道数的输入
                kernel_size = 7; padding = 3; stride = 2 # ​​Stem 模式​​（is_stem=True）：采用 7x7 大核卷积（步长 2），初始阶段快速降维。
            else:
                kernel_size = 3; padding = 1; stride = 2 # 非 Stem 模式​​：3x3 卷积（步长 2），渐进式特征提取。
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)                    
        
        else:# False：直接通过 patch_size 分块（如 4x4 无重叠块），类似 ViT
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim) # 启用 norm_layer​​：归一化后保持相同形状。
        else:
            self.norm = None  # 未启用 norm_layer​​：[B, embed_dim, H', W']（如 [1, 96, 56, 56]）。

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            # 若宽度 W 不能被 patch_size[1] 整除，则在 ​​右侧​​ 填充 0，使新宽度为 W + (patch_size[1] - W % patch_size[1])
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            #若高度 H 不能被 patch_size[0] 整除，则在 ​​底部​​ 填充 0，使新高度为 H + (patch_size[0] - H % patch_size[0])。
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww 通过卷积层 (self.proj) 将图像分块并映射到嵌入空间。
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            # 从第 2 个维度（索引从 0 开始）开始展平，即保留前两个维度（batch_size 和 channels），将后两个维度（height 和 width）合并为一个维度。
            x = x.flatten(2).transpose(1, 2) # 展平空间维度​​：将投影后的特征图 [B, C, H', W'] 转换为 [B, C, H'W']，再转置为 [B, H'W', C]
            x = self.norm(x) # 在通道维度 C 上应用 norm_layer（如 LayerNorm）
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # ​​恢复形状​​：转置并重塑回 [B, C, H', W']，保留空间结构。
        return x




# ​​类名​​：FocalNet
# ​​"Focal"​​：表明网络采用 ​​多尺度焦点调制技术​​（Focal Modulation），通过动态调整不同区域的特征重要性来高效建模长距离依赖。
# ​​"Net"​​：表示这是一个完整的 ​​主干网络架构​​，用于图像特征提取。
# ​​定位​​：一种基于 ​​焦点调制模块​​ 的视觉主干网络，旨在替代传统 CNN 或 Transformer，平衡计算效率与建模能力。
class FocalNet(nn.Module):
    """ FocalNet backbone.

    Args:
        pretrain_img_size (int):        预训练模型的输入图像尺寸（用于绝对位置编码）. Default 224.
        patch_size (int | tuple(int)):  图像分块大小. Default: 4.
        in_chans (int):                 输入图像通道数. Default: 3.
        embed_dim (int):                线性投影输出通道数. Default: 96.
        depths (tuple[int]):            Swin Transformer各阶段的层数.
        mlp_ratio (float):              MLP隐藏层维度与嵌入维度的比值. Default: 4.
        drop_rate (float):              普通Dropout概率.
        drop_path_rate (float):         随机深度丢弃率 Default: 0.2.
        norm_layer (nn.Module):         归一化层类型. Default: nn.LayerNorm.
        patch_norm (bool):              是否在分块嵌入后添加归一化. Default: True.
        out_indices (Sequence[int]):    指定输出哪些阶段的特征.
        frozen_stages (int):            冻结训练的阶段数（-1表示不冻结）.
        focal_levels (Sequence[int]):   四阶段中每阶段的多焦点层级数
        focal_windows (Sequence[int]):  四阶段中首层焦点窗口大小
        use_conv_embed (bool):          是否使用重叠卷积进行分块嵌入
        use_checkpoint (bool):          是否使用梯度检查点节省显存. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=1600,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.3, # 0.3 or 0.4 works better for large+ models
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 focal_levels=[3,3,3,3], 
                 focal_windows=[3,3,3,3],
                 use_conv_embed=False, 
                 use_postln=False, 
                 use_postln_in_modulation=False, 
                 use_layerscale=False, 
                 normalize_modulator=False, 
                 use_checkpoint=False,                  
        ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches 
        # 将输入图像分割为块 在通道堆叠，输出形状 [B, C' , H', W']
        self.patch_embed = PatchEmbed( 
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, 
            use_conv_embed=use_conv_embed, is_stem=True)

        #​ ​位置随机丢弃（PosDrop）
        # 每个输入张量的元素（或神经元输出）以概率 p 被​​独立地置为0​​，否则保留原值
        # 未被丢弃的元素会按 1/(1-p) 放大（训练阶段），以保持训练和测试时的总激活强度一致。
        self.pos_drop = nn.Dropout(p=drop_rate) 

        # stochastic depth ​​层级构建（BasicLayer）​​
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer( # 每层 BasicLayer类 包含多个 FocalModulationBlock
                dim=int(embed_dim * 2 ** i_layer),# 维度逐层翻倍（embed_dim * 2**i_layer）
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,#最后一层不降采样（downsample=None），保留分辨率。
                focal_window=focal_windows[i_layer], 
                focal_level=focal_levels[i_layer], 
                use_conv_embed=use_conv_embed,
                use_postln=use_postln, 
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator, 
                use_layerscale=use_layerscale, 
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # num_features 是一个列表
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    # 冻结指定阶段的参数（如 frozen_stages=2 冻结前两层的权重），用于迁移学习或部分微调
    def _freeze_stages(self): 
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x_emb = self.patch_embed(x) # # 分块嵌入 → [B, C', H', W']
        Wh, Ww = x_emb.size(2), x_emb.size(3)   

        x = x_emb.flatten(2).transpose(1, 2)    # 序列化为 [B, H'W', C]
        x = self.pos_drop(x)    # # 随机丢弃

        # 每层 BasicLayer类 包含多个 FocalModulationBlock，通道数逐层翻倍
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)   # 逐层处理       

            if i in self.out_indices: 
            # 在模型初始化时，通过以下代码为每个输出阶段添加归一化层    
                norm_layer = getattr(self, f'norm{i}') 
                x_out = norm_layer(x_out) # 归一化并输出
            # 每层输出特征被归一化并重塑为 [B, C, H, W]，存入 outs 列表。                  
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs, x_emb # 多尺度特征 + 初始块


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(FocalNet, self).train(mode)
        self._freeze_stages()


# FocalNet 模型工厂函数​​，根据指定的模型名称（如 focalnet_L_384_22k）和可选参数，
# 快速构建预定义配置的 FocalNet 主干网络。
# 其核心作用是 ​​统一管理不同规模的模型配置​​，简化模型实例化流程。
# 这个函数是 ​​模型工厂​​，类似于“菜单”，根据你选择的模型名称（如 focalnet_L_384_22k），
# 自动配置好对应的网络参数，然后组装成一个完整的 FocalNet 模型。
# 类似于去餐厅点菜，告诉服务员你要“套餐A”，厨师就会按固定配方做菜。
def build_focalnet(modelname, **kw):

    assert modelname in [ # 确保 modelname 是预定义的合法名称，防止无效配置。
        'focalnet_L_384_22k', 
        'focalnet_L_384_22k_fl4', 
        'focalnet_XL_384_22k', 
        'focalnet_XL_384_22k_fl4', 
        'focalnet_H_224_22k', 
        'focalnet_H_224_22k_fl4',         
        ]
    
    # 若用户指定了 focal_levels 或 focal_windows 为单个值，
    # 将其扩展为 4 个元素的列表（对应模型的四个阶段）。
    if 'focal_levels' in kw:
        kw['focal_levels'] = [kw['focal_levels']] * 4

    if 'focal_windows' in kw:
        kw['focal_windows'] = [kw['focal_windows']] * 4

    model_para_dict = {
        'focalnet_L_384_22k': dict(
            embed_dim=192, # 初始嵌入维度（逐层翻倍）。
            depths=[ 2, 2, 18, 2 ], # 各阶段的块数（如 [2,2,18,2] 表示第三阶段有 18 个块）。
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]),  # 各阶段的焦点级别数（控制多尺度聚合范围）。
            focal_windows=kw.get('focal_windows', [5, 5, 5, 5]), # 各阶段的初始窗口大小（影响局部感受野）。
            use_conv_embed=True, # 是否使用卷积嵌入（替代直接分块）。
            use_postln=True,    # 是否在后处理中使用 LayerNorm。
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),
        'focalnet_L_384_22k_fl4': dict(
            embed_dim=192,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [4, 4, 4, 4]), 
            focal_windows=kw.get('focal_windows', [3, 3, 3, 3]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=True, 
        ),
        'focalnet_XL_384_22k': dict(
            embed_dim=256,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]), 
            focal_windows=kw.get('focal_windows', [5, 5, 5, 5]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),   
        'focalnet_XL_384_22k_fl4': dict(
            embed_dim=256,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [4, 4, 4, 4]), 
            focal_windows=kw.get('focal_windows', [3, 3, 3, 3]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=False, 
            use_layerscale=True, 
            normalize_modulator=True, 
        ),           
        'focalnet_H_224_22k': dict(
            embed_dim=352,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [3, 3, 3, 3]), 
            focal_windows=kw.get('focal_windows', [3, 3, 3, 3]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_layerscale=True, 
            use_postln_in_modulation=True, 
            normalize_modulator=False, 
        ),   
        'focalnet_H_224_22k_fl4': dict(
            embed_dim=352,
            depths=[ 2, 2, 18, 2 ],
            focal_levels=kw.get('focal_levels', [4, 4, 4, 4]), 
            focal_windows=kw.get('focal_windows', [3, 3, 3, 3]), 
            use_conv_embed=True, 
            use_postln=True, 
            use_postln_in_modulation=True, 
            use_layerscale=True, 
            normalize_modulator=False, 
        ),                        
    }

    kw_cgf = model_para_dict[modelname] # 获取预配置
    kw_cgf.update(kw) # 合并用户自定义参数
    model = FocalNet(**kw_cgf) # 组装模型 
    return model


# [FocalNet 整体架构]
# Input (B, 3, H, W)
# ├─ PatchEmbed ───▶ [B, C, H/4, W/4] → x_emb
# │   (4x4卷积嵌入，步长4)
# │
# └─ 分层处理 (共num_layers个阶段)
#    ├─ Stage 0 (dim=C, 分辨率H/4×W/4)
#    │   ├─ BasicLayer
#    │   │   ├─ ×depth[0] FocalModulationBlock （depth是多少 每个layer不一样 精炼特征）
#    │   │   │   ├─ PreNorm
#    │   │   │   ├─ FocalModulation (含 focal_level 级卷积)
#    │   │   │   │   ├─ 线性投影 → q/ctx/gates
#    │   │   │   │   ├─ 多尺度卷积堆 (核尺寸: focal_window + k*focal_factor)
#    │   │   │   │   └─ 门控加权融合
#    │   │   │   ├─ MLP扩展
#    │   │   │   └─ 残差连接×2
#    │   │   ├─ ×depth[1] FocalModulationBlock
#    │   │   │   ├─ PreNorm
#    │   │   │   ├─ FocalModulation (含 focal_level 级卷积)
#    │   │   │   │   ├─ 线性投影 → q/ctx/gates
#    │   │   │   │   ├─ 多尺度卷积堆 (核尺寸: focal_window + k*focal_factor)
#    │   │   │   │   └─ 门控加权融合
#    │   │   │   ├─ MLP扩展
#    │   │   │   └─ 残差连接×2
#    │   │   ├─ ×depth[2] FocalModulationBlock
#    │   │   │   ├─ PreNorm
#    │   │   │   ├─ FocalModulation (含 focal_level 级卷积)
#    │   │   │   │   ├─ 线性投影 → q/ctx/gates
#    │   │   │   │   ├─ 多尺度卷积堆 (核尺寸: focal_window + k*focal_factor)
#    │   │   │   │   └─ 门控加权融合
#    │   │   │   ├─ MLP扩展
#    │   │   │   └─ 残差连接×2
#    │   │   ├─ ×depth[3] FocalModulationBlock
#    │   │   │   ├─ PreNorm
#    │   │   │   ├─ FocalModulation (含 focal_level 级卷积)
#    │   │   │   │   ├─ 线性投影 → q/ctx/gates
#    │   │   │   │   ├─ 多尺度卷积堆 (核尺寸: focal_window + k*focal_factor)
#    │   │   │   │   └─ 门控加权融合
#    │   │   │   ├─ MLP扩展
#    │   │   │   └─ 残差连接×2
#    │   │   └─ 2x下采样 → [B, 2C, H/8, W/8]
#    │   └─ 输出归一化 (若在out_indices)
#    │
#    ├─ Stage 1 (dim=2C, 分辨率H/8×W/8)
#    │   ┌─ 结构同上，深度depth[1]，焦点窗口调整...
#    │
#    └─ Stage N (dim=2^N*C, 分辨率H/(4 * 2^N)×W/(4 * 2^N))
#        └─ 无下采样保持最终分辨率

# Outputs:
# ├─ outs: 多尺度特征列表 [(B, C_i, H_i, W_i)] 
# └─ x_emb: 初始嵌入特征

# 每Stage分辨率减半（H/4 → H/8 → ...）
# 每Stage通道数翻倍（C → 2C → 4C...）

# FocalModulationBlock的q ctx gates是什么？
# 输入 x 的形状​​：假设为 (B, C_total, H, W)，其中 C_total = C + C + (self.focal_level+1)。
# q：形状 (B, C, H, W)，表示查询（Query）特征。通过线性变换从输入中提取 ​​与位置强相关​​ 的特征
# ctx：形状 (B, C, H, W)，表示上下文（Context）特征。通过后续的 ​​多尺度卷积​​ 和 ​​门控融合​​ 提取上下文
# gates：形状 (B, self.focal_level+1, H, W)，表示多焦点级别的门控权重。动态调节不同尺度特征的贡献权重
# ​尽管 q、ctx、gates 来自x经过同一线性层映射出来的三个区域，但模型可以通过反向传播自动学习如何将输入特征分解为三个功能不同的部分。
   

# debug搞清网络行为

# 每个模块清楚梳理 每个类之间的关系网络 每个类的输入输出size 一些关键操作的意义