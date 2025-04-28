# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath



# 类名​​：LayerNormFunction
# ​​"LayerNorm"​​：表示该类实现了 ​​层归一化（Layer Normalization）​​ 的核心计算。
# ​​"Function"​​：继承自 PyTorch 的 torch.autograd.Function，说明这是自定义的自动微分函数，用于手动实现前向和反向传播逻辑。
# ​​定位​​：一种针对 ​​图像数据（2D/4D 张量）​​ 的层归一化操作，支持自定义梯度计算，适用于需要优化归一化过程的场景。
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):# 输入形状：[B, C, H, W]
        # ctx​​：PyTorch 自定义函数中用于保存前向传播的上下文对象（Context），用于传递前向计算的中间结果到反向传播。
        # ​eps​​：一个极小的正数（如 10−5），用于数值稳定性，防止分母为零。
        ctx.eps = eps
        N, C, H, W = x.size()# 输入形状：[B, C, H, W]
        mu = x.mean(1, keepdim=True)# 计算通道均值 [B,1,H,W]
        var = (x - mu).pow(2).mean(1, keepdim=True)# 计算通道方差 [B,1,H,W]
        y = (x - mu) / (var + eps).sqrt()# 归一化
        ctx.save_for_backward(y, var, weight)# 保存中间变量用于反向传播
        # 在归一化操作中，​​仿射变换​​ 指对归一化后的数据进行 ​​缩放（Scale）​​ 和 ​​平移（Shift）​​ 的线性变换
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)# 仿射变换
        return y

    @staticmethod
    def backward(ctx, grad_output):# grad_output 形状：[B, C, H, W]
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1) # 梯度乘以权重
        mean_g = g.mean(dim=1, keepdim=True)# 梯度均值 [B,1,H,W]

        mean_gy = (g * y).mean(dim=1, keepdim=True)# 梯度与 y 的乘积均值
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)# 输入梯度
        # grad_weight = (grad_output * y).sum([0,2,3])  # 权重梯度 [C]
        # grad_bias = grad_output.sum([0,2,3])          # 偏置梯度 [C]    
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


# ​​类名​​：LayerNorm2d
# ​​"LayerNorm"​​：表示这是一个 ​​层归一化（Layer Normalization）​​ 模块，沿通道维度（C）对每个样本进行归一化。
# ​​"2d"​​：表示该模块专为 ​​二维数据（如图像）​​ 设计，输入形状为 [B, C, H, W]（批次、通道、高度、宽度）。
# ​​定位​​：一种针对 ​​图像数据（4D 张量）​​ 的层归一化模块，封装了归一化计算和可学习参数。
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# SimpleGate
# ​​"Simple"​​：表示这是一个 ​​简化版的门控机制​​，结构简单，无额外参数。
# ​​"Gate"​​：表明其功能是通过 ​​门控操作控制信息流动​​，类似生物神经元的激活与抑制。
# ​​定位​​：一种轻量级的特征交互模块，通过分割输入通道并逐元素相乘，实现特征选择与非线性增强。
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1) # 将输入张量沿通道维度（dim=1）均分为两部分 x1 和 x2，并对它们进行 ​​逐元素相乘​​
        return x1 * x2


# "NAF"​​：通常为 ​​Non-linear Activation Free​​ 的缩写，表明该模块通过 ​​乘法操作​​ 替代传统激活函数（如ReLU），减少显式非线性层的使用。
# ​​"Block"​​：表示这是一个基础网络模块，可堆叠构建深层网络。
class NAFBlock(nn.Module):
    def __init__(self, dim, expand_dim, out_dim, kernel_size=3, layer_scale_init_value=1e-6, drop_path=0.):
        super().__init__()
        drop_out_rate = 0. 
        dw_channel = expand_dim
        # 1x1升维
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # 深度可分离卷积
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        # 1x1降维至原通道数
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention 简化通道注意力 (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True), # 通道注意力权重生成
        )

        # SimpleGate
        self.sg = SimpleGate() # 通道分割与乘积

        ffn_channel = expand_dim
        # 前馈分支
        self.conv4 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True) # 1x1升维
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=out_dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True) # 1x1调整输出维度


        self.norm1 = LayerNorm2d(dim) # 通道归一化
        self.norm2 = LayerNorm2d(dim)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity() # 随机丢弃 
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 层缩放参数
        self.beta = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True) # 初始化缩放因子
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)) * layer_scale_init_value, requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)  # 归一化

        x = self.conv1(x)   # 升维 [B, dim, H, W] → [B, expand_dim, H, W]
        x = self.conv2(x)   # 深度可分离卷积（空间特征提取）
        x = self.sg(x)      # 通道分割 → [B, expand_dim//2, H, W] ×2，乘积后输出同维度
        x = x * self.sca(x) # 通道注意力加权
        x = self.conv3(x)   #  降维至原通道数 [B, dim, H, W]

        x = self.dropout1(x) 

        y = inp + x * self.beta # 残差连接（层缩放）

        x = self.conv4(self.norm2(y))  # 前馈分支升维
        x = self.sg(x)                  # 门控交互 
        x = self.conv5(x)               # 调整输出维度

        x = self.dropout2(x)

        return y + x * self.gamma       # 最终残差输出


class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(  
                                        # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
                                        nn.Linear(inchannel, outchannel),
                                        LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        self.upsample  = nn.Upsample(scale_factor=2**ratio, mode='bilinear')
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x = x.permute(0, 3, 1, 2)
        
        return self.upsample(x)

# 这是一个 ​​支持多种数据格式的层归一化（Layer Normalization）模块​​，允许处理 channels_first
# （如 PyTorch 默认的 [B, C, H, W]）
# 和 channels_last（如 TensorFlow 的 [B, H, W, C]）两种输入格式，适用于图像或其他多维数据。
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first", elementwise_affine = True):
        super().__init__()
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last": # channels_last​​：直接调用 PyTorch 内置的 F.layer_norm。
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first": # channels_first​​：手动沿通道维度（dim=1）计算均值与方差，实现归一化。
            u = x.mean(1, keepdim=True)         # keepdim=True​​： 保持输出张量的维度与原输入一致。
            s = (x - u).pow(2).mean(1, keepdim=True) # .pow(2)​​：对零中心化后的数据逐元素取平方（等价于 (x - u) ** 2）
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.elementwise_affine:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x






# ​​"ConvNext"​​：表明该模块属于 ​​ConvNeXt 架构​​，一种借鉴 Transformer 设计思想的现代卷积网络。
# ​​"Block"​​：表示这是网络的基本构建单元，类似 ResNet 的残差块，但结构优化。
class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()

        # 深度卷积（仅空间特征提取）
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel) # depthwise conv
        # 层归一化（channels_last 模式）
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        # 1x1卷积（通过线性层实现）
        self.pwconv1 = nn.Linear(in_channel, hidden_dim) # 升维# pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)# 降维
        # 层缩放参数
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # 随机路径丢弃
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # 深度卷积 [B, C, H, W]
        x = x.permute(0, 2, 3, 1) #转为 channels_last [B, H, W, C] # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)   # 层归一化
        x = self.pwconv1(x)    # 升维至 hidden_dim
        x = self.act(x)          # GELU 激活
        x = self.pwconv2(x)      # 降维至 out_channel
        if self.gamma is not None:   
            x = self.gamma * x  # 层缩放
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W) # 转回 channels_first [B, C, H, W]

        x = input + self.drop_path(x)   # 残差连接 + DropPath
        return x








# ​​"Decoder"​​：表示这是一个​​解码器模块​​，负责将低分辨率、高维的特征图逐步上采样并恢复为高分辨率图像，常见于图像重建任务（如超分辨率、图像修复）。
# ​Decoder 是一个 ​​多阶段特征融合的解码器模块​​，通过上采样、通道调整与残差块堆叠，将编码器的深层特征逐步重建为高分辨率图像。
# 其名称直接反映了功能本质（解码），适用于需要从低维特征恢复高分辨率像素的任务（如超分辨率、图像修复）
class Decoder(nn.Module):
    def __init__(self, depth=[2,2,2,2], dim=[112, 72, 40, 24], block_type = None, kernel_size = 3) -> None:
        super().__init__()
        self.depth = depth # 各阶段模块的堆叠次数（如每个阶段用2个block）
        self.dim = dim       # 各阶段的通道维度（如[112, 72, 40, 24]）
        self.block_type = block_type     # 核心模块类型（如ConvNextBlock）
        self._build_decode_layer(dim, depth, kernel_size)  # 核心模块类型（如ConvNextBlock）
        self.pixelshuffle=nn.PixelShuffle(2) # 上采样2倍（输出通道3，分辨率翻倍）
        # self.star_relu=StarReLU()
        self.projback_ = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=2 ** 2 * 3 , kernel_size=1), # 1x1卷积调整通道
            nn.PixelShuffle(2) # 上采样2倍（输出通道3，分辨率翻倍）
        )
        self.projback_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=dim[-1],
                out_channels=2 ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(2)
        )
        
    def _build_decode_layer(self, dim, depth, kernel_size):
        normal_layers = nn.ModuleList() # 各阶段的特征处理模块（block_type）
        upsample_layers = nn.ModuleList() # 上采样层（双线性插值）
        proj_layers = nn.ModuleList()  # 投影层（通道调整 + 归一化 + 激活）

        norm_layer = LayerNorm # 自定义的层归一化

        # 构建每个阶段的模块
        for i in range(1, len(dim)):
            # 特征处理模块（如堆叠多个block_type）
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            
             # 上采样层（放大2倍）
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            
            # 投影层：1x1卷积调整通道 + 归一化 + 激活
            proj_layers.append(nn.Sequential(
                nn.Conv2d(dim[i-1], dim[i], 1, 1), 
                norm_layer(dim[i]),
                # StarReLU() #self.star_relu()
                nn.GELU()
                ))
        
        for i in range(1, len(dim)):
            module = [self.block_type(dim[i], dim[i], dim[i], kernel_size) for _ in range(depth[i])]
            normal_layers.append(nn.Sequential(*module))
            upsample_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            proj_layers.append(nn.Sequential(
                               nn.Conv2d(dim[i-1], dim[i], 1, 1),
                               norm_layer(dim[i]),
            ))

        # 保存到类属性    
        self.normal_layers = normal_layers
        self.upsample_layers = upsample_layers
        self.proj_layers = proj_layers

    def _forward_stage(self, stage, x):
        x = self.proj_layers[stage](x) # 1x1 卷积调整通道数，LayerNorm 稳定训练，GELU 引入非线性。
        x = self.upsample_layers[stage](x) # 使用 nn.Upsample（双线性插值）和 PixelShuffle（子像素卷积）逐步提升分辨率。
        return self.normal_layers[stage](x)

    # c3, c2, c1, c0 通常是编码器不同层级的输出（分辨率从低到高，通道数从多到少）
    # ​​融合方式​​：每个解码阶段处理深层特征（如c3），上采样后与浅层特征（如c2）逐元素相乘（*），保留细节信息
    def forward(self, c3, c2, c1, c0):
        # 分离clean和ref路径的输入特征（假设双路径处理）
        c0_clean, c0_ref = c0, c0 
        c1_clean, c1_ref = c1, c1 
        c2_clean, c2_ref = c2, c2 
        c3_clean, c3_ref = c3, c3 

        # Clean路径处理 ​Clean路径​​：处理待修复图像的特征。
        x_clean = self._forward_stage(0, c3_clean) * c2_clean
        x_clean = self._forward_stage(1, x_clean) * c1_clean
        x_clean = self._forward_stage(2, x_clean) * c0_clean
        x_clean = self.projback_(x_clean) # 最终投影为RGB图像
        
        # Ref路径处理（类似clean路径） Ref路径​​：处理参考图像的特征（如类似场景的高质量图像）
        x_ref = self._forward_stage(3, c3_ref) * c2_ref
        x_ref = self._forward_stage(4, x_ref) * c1_ref
        x_ref = self._forward_stage(5, x_ref) * c0_ref
        x_ref = self.projback_2(x_ref)

        # 合并双路径输出
        x=torch.cat((x_clean,x_ref),dim=1)
        return x







# SimDecoder
# ​​"Sim"​​：代表 "Simple"，表明这是一个​​简化的解码器模块​​，结构轻量。
# ​​"Decoder"​​：表示功能是将低分辨率特征图解码为高分辨率图像。
# ​​功能定位​​：通过极简的层归一化、1x1卷积和子像素上采样（PixelShuffle），将编码器的深层特征直接转换为目标图像。
class SimDecoder(nn.Module):
    def __init__(self, in_channel, encoder_stride) -> None:
        super().__init__()
        self.projback = nn.Sequential(
            LayerNorm(in_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),# 调整通道数为 encoder_stride² * 3，为后续上采样做准备
            nn.PixelShuffle(encoder_stride),# 通过子像素操作将通道数转换为空间分辨率，实现高效上采样
        )

    def forward(self, c3):
        return self.projback(c3)
    

# StarReLU
# ​​"Star"​​：可能暗示其性能优越性（如“明星”激活函数），或指代其数学形式中的平方项（x² 形似星形标记）。
# ​​"ReLU"​​：表明它是 ReLU 的改进变体，保留 ReLU 的核心特性（单侧抑制）。
# ​​功能定位​​：一种 ​​带可学习参数的非线性激活函数​​，通过引入缩放和平移参数增强模型表达能力。
#StarReLU 是一种 ​​改进的 ReLU 类激活函数​​，通过可学习的缩放和偏置参数，结合平方非线性，平衡了简单性、表达能力和训练稳定性。其名称反映了“星形”平方操作与 ReLU 的结合，适用于需要增强模型非线性或自适应调整激活分布的深度学习任务。
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=True):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias
    
    # ​​scale 的作用​​：若学习到较大的值，会放大重要特征的贡献
    # bias 的作用​​：正偏置可避免激活值全为0（如深层网络的梯度消失问题）。

