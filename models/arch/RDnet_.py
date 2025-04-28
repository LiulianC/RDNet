import numpy as np
from models.arch.focalnet import build_focalnet
import torch
import torch.nn as nn
from models.arch.modules_sig import ConvNextBlock, Decoder, LayerNorm, NAFBlock, SimDecoder, UpSampleConvnext
from models.arch.reverse_function import ReverseFunction
from timm.models.layers import trunc_normal_


# Fusion 是一个用于 ​​多尺度特征融合的模块​​，其主要功能是将来自不同层级的特征（如上采样和下采样特征）进行融合，
# 常用于类似 U-Net 的分层架构或特征金字塔网络（FPN）。
# 通过 level 参数控制不同阶段的融合策略，支持 ​​纯下采样（特定层级）​​ 和 ​​上采样+下采样融合（其他层级）​​。

class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None: # channels:各层级的通道数列表（如 [64, 128, 256, 512]）
        super().__init__()

        self.level = level # 当前模块所在的层级编号（通常范围为 0–3，层级越高特征图分辨率越低）
        self.first_col = first_col # 标记是否为第一列的首个模块（决定是否跳过上采样的特征融合）
        self.down = nn.Sequential( # 下采样模块
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col: # 上采样模块（UpSampleConvnext）
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()
    
    def forward(self, *args):
                            # c_up（​​可选​​）：来自较高层级（分辨率较低）的特征图（当 first_col=False 时可能存在）
        c_down, c_up = args # c_down（​​必选​​）：来自较低层级（分辨率较高）的特征图（形状 [B, C, H, W]）
        channels_dowm=c_down.size(1)
        if self.first_col: # 首列模式
            x_clean = self.down(c_down) # 仅下采样（但最高层级通常无更高层级特征）
            return x_clean
        if c_up is not None:
            channels_up=c_up.size(1)
        if self.level == 3:
            x_clean = self.down(c_down) # 仅下采样（但最高层级通常无更高层级特征）
        else:
            x_clean = self.up(c_up) + self.down(c_down) # 中间层级 (level ∈ [0,1,2]) 上采样 + 下采样，特征相加融合
            
        return x_clean  # 最终融合后的特征图 x_clean，分辨率与当前层级一致。






# Level 是用于构建 ​​分层神经网络（如 U-Net、FPN）​​ 的核心模块，负责处理特定层级的特征。
class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, block_type=ConvNextBlock) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col) # 特征融合模块，根据 level 和 first_col 选择是否融合上/下层特征
        modules = [block_type(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i]) for i in
                   range(layers[level])]
        self.blocks = nn.Sequential(*modules) # 由多个 ConvNextBlock 块组成的序列，用于深度特征提取。块数量由 layers[level] 决定

    def forward(self, *args):
        x = self.fusion(*args)
        x_clean = self.blocks(x)
        return x_clean

# level	            int	                当前模块所在的层级编号（如 0 表示最高分辨率层级）
# channels	        List[int]	        各层级的通道数列表（如 [64, 128, 256, 512]）
# layers	        List[int]	        各层级包含的 ConvNextBlock 块数（如 [3, 4, 6, 3]）
# kernel_size	    int	                ConvNextBlock 的卷积核大小
# first_col	        bool	            是否为第一列（如 U-Net 的编码器首列，仅下采样无特征融合）
# dp_rate	        List[float]	        每个块的随机丢弃路径概率（Stochastic Depth）
# block_type	    nn.Module	        块类型（默认 ConvNextBlock，可替换为其他模块）





# SubNet 是一个 ​​多层级可逆神经网络模块​​，支持两种运行模式：
# ​​常规模式​​：保存所有中间变量用于反向传播，显存占用较高但计算速度快。
# ​​内存优化模式​​：通过 ReverseFunction 实现按需重计算，显著减少显存占用，适用于大模型或高分辨率输入。
# 其核心功能是通过 ​​残差连接 + 多层级特征处理​​ 构建深度网络，常用于图像分割、生成等需要多尺度特征的任务。
class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, block_type=ConvNextBlock) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates, block_type=block_type)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        c0 = self.alpha0 * c0 + self.level0(x, c1)  # 层级0处理
        c1 = self.alpha1 * c1 + self.level1(c0, c2) # 层级1处理
        c2 = self.alpha2 * c2 + self.level2(c1, c3) # 层级2处理
        c3 = self.alpha3 * c3 + self.level3(c2, None) # 层级3处理
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):
        x, c0, c1, c2, c3 = args
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    # _clamp_abs 方法用于 ​​约束张量数据的绝对值下限​​，确保其绝对值不低于指定值，同时保持原始符号
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign() # ​​符号保留​​
            data.abs_().clamp_(value) # 将输入张量 data 的每个元素的绝对值限制在 [value, +∞) 范围内
            data *= sign













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

class FullNet_NLP(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5,loss_col=4, kernel_size=3, num_classes=1000,
                 drop_path=0.0, save_memory=True, inter_supv=True, head_init_scale=None, pretrained_cols=16) -> None:
        super().__init__()
        self.num_subnet = num_subnet
        self.Loss_col=(loss_col+1)
        self.inter_supv = inter_supv
        self.channels = channels
        self.layers = layers
        self.stem_comp = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=5, stride=2, padding=2),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )
        self.prompt=nn.Sequential(nn.Linear(in_features=6,out_features=512),
                                  StarReLU(),
                                  nn.Linear(in_features=512,out_features=channels[0]),
                                  StarReLU(),
                                  )
        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, 
                dp_rates=dp_rate, save_memory=save_memory,
                block_type=NAFBlock))

        channels.reverse()
        self.decoder_blocks = nn.ModuleList(
            [Decoder(depth=[1, 1, 1, 1], dim=channels, block_type=NAFBlock, kernel_size=3) for _ in
             range(3)])

        self.apply(self._init_weights)
        self.baseball = build_focalnet('focalnet_L_384_22k_fl4')
        self.baseball_adapter = nn.ModuleList()
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 2, 64 * 2, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 4, 64 * 4, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 8, 64 * 8, kernel_size=1))
        self.baseball.load_state_dict(torch.load('D:/gzm-RDNet/RDNet\models/pretrained/focal.pth'))
    def forward(self, x_in,alpha,prompt=True):
        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4

        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = self.baseball_adapter[1](c0),\
                         self.baseball_adapter[2](c1),\
                         self.baseball_adapter[3](c2),\
                         self.baseball_adapter[4](c3)
        if prompt==True:
            prompt_alpha=self.prompt(alpha)
            prompt_alpha = prompt_alpha.unsqueeze(-1).unsqueeze(-1)
            x=prompt_alpha*x_stem
        else :
            x = x_stem
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
            if i>(self.num_subnet-self.Loss_col):
                x_img_out.append(torch.cat([x_in, x_in], dim=-3) - self.decoder_blocks[-1](c3, c2, c1, c0) )
 
        return x_cls_out, x_img_out

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

   
