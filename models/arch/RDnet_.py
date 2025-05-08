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
        self.first_col = first_col # first_col=True 在FULLnet_NLP赋值 代表 当前是 第一层subnet
        self.down = nn.Sequential( # 下采样模块
            # level - 1：当前级（分辨率更高），[level]：下一级 下采样 通道变化，尺寸减半 level是0~3
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
        ) if level in [1, 2, 3] else nn.Identity() # 第0层没有上一级，所以不做任何操作

        if not first_col: # 上采样模块（UpSampleConvnext）
            self.up = UpSampleConvnext(1, channels[level + 1], channels[level]) if level in [0, 1, 2] else nn.Identity()
    
    def forward(self, *args):
        # c_down：当前level的上一级 需要下采样 所以叫down
        # c_up：当前level的下一级 需要上采样 所以叫up
        c_down, c_up = args 
        channels_dowm=c_down.size(1)

        # 如果当前是第0级level first_col=True 在FULLnet_NLP赋值 代表 当前是 第0级subnet
        if self.first_col: 
            x_clean = self.down(c_down) # 给第0level的下采样 也就是对分块化的x下采样
            return x_clean 
        
        if c_up is not None: # 如果当前level的下一级不为空
            channels_up=c_up.size(1)

        # 如果到了最高层级
        if self.level == 3: 
            x_clean = self.down(c_down) # x_clean等于下采样的
        else:
            # 中间层级 (level ∈ [1,2])的x_clean等于 上采样 + 下采样，特征相加融合
            # level==0时会不会也在这里产生x_clean?不会 因为在第一个if里已经return了
            x_clean = self.up(c_up) + self.down(c_down) 
            


        # level0 : x_clean = self.down(c_down)
        # level1~2 : x_clean = self.up(c_up) + self.down(c_down)    
        # level3 : x_clean = self.down(c_down)
        return x_clean  






# Level 是用于构建 ​​分层神经网络（如 U-Net、FPN）​​ 的核心模块，负责处理特定层级的特征。

# level参数
# level 由SubNet定义level的时候传入0 1 2 3
# channel [32, 64, 96, 128]
# layers=[2, 3, 6, 3]
# kernel_size=3
# block_type=NAFBlock
# first_col 在本网络决定
class Level(nn.Module):
    # level 和 layers 区别：level 是 fullnet_NLP的第几层subnet； layer是一个level有多少个ConvNextBlock
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, block_type=ConvNextBlock) -> None: # block_type被赋值为 NAFBlock
        super().__init__()

        # layers 是一个列表，表示每个层级的块数（如 [3, 4, 6, 3]），这里sum(layers[:level])表示当前层级之前的所有块数之和
        countlayer = sum(layers[:level])
        # 隐藏层通道数=expansion*in_channels
        expansion = 4
        self.fusion = Fusion(level, channels, first_col) # 特征融合模块，根据 level 和 first_col 选择是否融合上/下层特征
        
        # ConvNextBlock(in_channels=当前level的通道数, hidden_dim = 4*in channel, 
        # out_channels=当前level通道数, 卷积核大小, 层级缩放系数初始化值, 随机丢弃路径概率)
        # # 多次经过 ConvNextBlock 有什么作用？​通过多个 ConvNextBlock 的连续作用，逐步增强特征的抽象能力
        modules = [block_type(channels[level], expansion * channels[level], channels[level], kernel_size=kernel_size,
                                 layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer + i])  # drop_path=dp_rate[countlayer + i] 得到第几level的第i个block的drop概率
                                 for i in range(layers[level])]
        
        # Sequential 将多个子模块按顺序串联成一个处理流水线​ x = block3(block2(block1(x_input)))
        # *modules 前面的星号（*）是 ​​解包操作符（Unpacking Operator）​​，它的作用是将一个列表（或元组）中的元素 ​​逐个展开​​，作为独立的参数传递给函数或构造函数。
        self.blocks = nn.Sequential(*modules) # 由多个 ConvNextBlock 块组成的序列，用于深度特征提取。块数量由 layers[level] 决定

    def forward(self, *args):  # args ： level0(x, c1) 只给俩参数
        x = self.fusion(*args) # 得到了Fusion的输出 x_clean
        x = self.blocks(x) # # 由多个 ConvNextBlock 块组成的序列，用于深度特征提取。块数量由 layers[level] 决定
        return x

# level	            int	                当前模块所在的层级编号（如 0 表示最高分辨率层级）
# channels	        List[int]	        各层级的通道数列表（如 [64, 128, 256, 512]）
# layers	        List[int]	        各层级包含的 ConvNextBlock 块数（如 [3, 4, 6, 3]）
# kernel_size	    int	                ConvNextBlock 的卷积核大小
# first_col	        bool	            first_col=True 在FULLnet_NLP赋值 代表 当前是 第0层subnet
# dp_rate	        List[float]	        每个块的随机丢弃路径概率（Stochastic Depth）
# block_type	    nn.Module	        块类型（默认 ConvNextBlock，可替换为其他模块）





# SubNet 是一个 ​​多层级可逆神经网络模块​​，支持两种运行模式：
# ​​常规模式​​：保存所有中间变量用于反向传播，显存占用较高但计算速度快。
# ​​内存优化模式​​：通过 ReverseFunction 实现按需重计算，显著减少显存占用，适用于大模型或高分辨率输入。
# 其核心功能是通过 ​​残差连接 + 多层级特征处理​​ 构建深度网络，常用于图像分割、生成等需要多尺度特征的任务。

# subnet的参数
# channel [32, 64, 96, 128]
# layers=[2, 3, 6, 3]
# kernel_size=3
# block_type=NAFBlock
# first_col 在本网络决定 是不是第0 level
class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, block_type=ConvNextBlock) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        # 创建一个形状为 (1, channels[3], 1, 1) 的张量，初始值为 shortcut_scale_init_value，并注册为模型的参数（nn.Parameter）。
        # 如果 shortcut_scale_init_value > 0，则启用该参数；否则设为 None（即不使用缩放）。
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
        
        # args就是之前的多尺度特征和块嵌入 x c0 c1 c2 c3
        x, c0, c1, c2, c3 = args 
        
        # 四个level 分别进行处理
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args): # args = x, c0, c1, c2, c3
        self._clamp_abs(self.alpha0.data, 1e-3) # .data返回张量的纯数值部分（剥离梯度计算相关的上下文），相当于获取参数的“原始值”。
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








# 定义新的激活函数
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







# 这是一个基于PyTorch的自定义神经网络模型类，继承自nn.Module，
# 属于​​多任务学习模型​​，结合了​​图像特征提取​​、​​多尺度子网络处理​​和​​解码重构​​功能。
# FullNet_NLP是一个结合预训练模型、多级子网络和条件提示机制的高级图像处理模型，
# 适用于需要多尺度特征融合和外部条件控制的任务（如条件图像恢复）
class FullNet_NLP(nn.Module): # 调用的时候输入参数：output_i, output_j = self.net_i(input_i,ipt,prompt=True)
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5,loss_col=4, kernel_size=3, num_classes=1000,
                 drop_path=0.0, save_memory=True, inter_supv=True, head_init_scale=None, pretrained_cols=16) -> None:
        super().__init__()
        self.num_subnet = num_subnet    # 子网络数量（默认5个）
        self.Loss_col=(loss_col+1)      # 控制损失计算起始层（从后往前数）
        self.inter_supv = inter_supv    # 是否启用中间监督。
        self.channels = channels        # 各子网络通道数配置如[32,64,96,128]
        self.layers = layers            # 各子网络层数配置（如[2,3,6,3]）
        # pretrained_cols：预训练模型通道数适配参数。
        self.stem_comp = nn.Sequential( # stem_comp​​：输入预处理模块，包含5x5卷积和通道归一化
            nn.Conv2d(3, channels[0], kernel_size=5, stride=2, padding=2),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )
        # prompt ​​：条件提示生成器，将PretrainedConvNext学到的特征 映射到特征空间，动态调整初始特征
        self.prompt=nn.Sequential(nn.Linear(in_features=6,out_features=512),
                                  StarReLU(),
                                  nn.Linear(in_features=512,out_features=channels[0]),
                                  StarReLU(),
                                  )
        
        # torch.linspace(0, drop_path, sum(layers))​​ 生成一个从 0 到 drop_path 的一维张量，包含 sum(layers) 个等间距值
        # x.item() 将 PyTorch 张量中的标量值转换为 Python 原生浮点数。张量 tensor(0.3) → 0.3
        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]

        # SubNet子网络​​：通过循环创建多个子网络，每个子网络处理不同层次的特征
        for i in range(num_subnet):

            # first_col=True 代表 当前是 第0层subnet
            first_col = True if i == 0 else False  
            
            # subnet的参数
            # channel [32, 64, 96, 128]
            # layers=[2, 3, 6, 3]
            # kernel_size=3
            # block_type=NAFBlock
            # first_col 在本网络决定
            self.add_module(f'subnet{str(i)}', SubNet(
                channels, layers, kernel_size, first_col, 
                dp_rates=dp_rate, save_memory=save_memory,
                block_type=NAFBlock))

        channels.reverse() # 将 [32, 64, 96, 128] → [128, 96, 64, 32] 供后续解码器使用

        # decoder_blocks​​：解码器模块列表，用于将子网络输出的多级特征重构为目标图像。
        self.decoder_blocks = nn.ModuleList(
            # depth：每层解码器使用NAFBlock的个数 block_type=NAFBlock
            # dim=[128, 96, 64, 32]
            [Decoder(depth=[1, 1, 1, 1], dim=channels, block_type=NAFBlock, kernel_size=3) for _ in
             range(3)]) # 0 1 2

        #baseball​​：加载预训练的FocalNet模型（focalnet_L_384_22k_fl4）作为特征提取器
        self.apply(self._init_weights)
        self.baseball = build_focalnet('focalnet_L_384_22k_fl4')

        # baseball_adapter​​：1x1卷积适配器，调整预训练模型输出通道以匹配子网络输入。
        self.baseball_adapter = nn.ModuleList()
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192, 64, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 2, 64 * 2, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 4, 64 * 4, kernel_size=1))
        self.baseball_adapter.append(nn.Conv2d(192 * 8, 64 * 8, kernel_size=1))
        self.baseball.load_state_dict(torch.load('D:/gzm-RDNet/RDNet\models/pretrained/focal.pth'))
    
    ## 调用的时候输入参数：output_i, output_j = self.net_i(input_i,ipt,prompt=True)
    # ipt 来自于cls 来自于 classifier.py中 PretrainedConvNext 的输出，size为(B,C=6)
    # ipt = self.net_c(input_i)
    def forward(self, x_in,alpha,prompt=True):
        # x_in：原始图像输入。alpha：条件向量（6维），用于生成提示特征.prompt：是否启用提示机制（默认开启）。
        x_cls_out = []
        x_img_out = []
        c0, c1, c2, c3 = 0, 0, 0, 0
        interval = self.num_subnet // 4 #  ​​计算一个间隔值​​，它将 self.num_subnet（子网的总数量）分成 4 个大致相等的部分。
        
        # ​特征提取​​：通过预训练FocalNet获取基础特征x_base和初始特征x_stem
        # x_base是focalnet提取的多尺度特征c0~c3分别是不同尺度的特征 x_stem是PatchEmbed分块的图片输入
        x_base, x_stem = self.baseball(x_in)
        c0, c1, c2, c3 = x_base
        # 通道适配​​：使用baseball_adapter调整c0~c3 stem的通道。
        x_stem = self.baseball_adapter[0](x_stem)
        c0, c1, c2, c3 = self.baseball_adapter[1](c0),\
                         self.baseball_adapter[2](c1),\
                         self.baseball_adapter[3](c2),\
                         self.baseball_adapter[4](c3)
        
        # 提示融合​​：若启用prompt，将alpha生成的提示向量与x_stem相乘，动态调制初始特征
        if prompt==True:
            # alpha 是clsmodel的 ipt，来自classifier.py中 PretrainedConvNext 的输出，size为(B,C=6)
            # PretrainedConvNext 是直接调用timm库的模型
            # alpha（前 3 维）：通道级缩放因子，增强重要特征。
            # beta（后 3 维）：通道级偏移量，补偿特征分布偏差。
            # 但是经过self.prompt,通道从6变成512再变成64
            prompt_alpha=self.prompt(alpha)
            
            # 在最后一个维度后添加一个维度（索引 -1 表示倒数第一）,再次在最后一个维度后添加一个维度.从 [B, C0] 调整为 [B, C0, 1, 1]
            # 使其能和四个维度的x_stem相乘
            prompt_alpha = prompt_alpha.unsqueeze(-1).unsqueeze(-1)
            x = prompt_alpha * x_stem # prompt_alpha size (B,64,1,1) x_stem size (B,64,H，W)
        else :
            x = x_stem

        # 子网络处理​​：循环调用各子网络（SubNet），传递并更新多级特征c0, c1, c2, c3
        # 子网络就是 把focalnet学到的各层次特征 在subnet仿射变换 在level的时候上下层交流融合
        for i in range(self.num_subnet): # i=0~3

            #getattr 用于​​动态获取对象的属性或方法​​。它的作用是通过字符串名称访问对象的成员
            # 每个循环都输入x, c0, c1, c2, c3 其中 c0, c1, c2, c3是上一个子网络的输出 所以 最后一个子网络才是集大成者！
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3) 
            
            # # 在本网络 num_subnet固定=4 和 Loss_col 固定=5 所以无论i是什么 都成立
            # 决定从第几个子网络（SubNet）开始 ​​计算损失​​ 这里给全部自网络都计算loss
            if i>(self.num_subnet-self.Loss_col): 

                # x_img_out 就是 原始图像在通道维度上拼接 - decoder输出
                # 为什么不是dim=1而是dim=-3?如果未来张量格式调整为其他顺序（如 (B, H, W, C)），使用 dim=-3 的代码会直接报错，提醒开发者检查维度逻辑，而 dim=1 会静默错误地拼接其他维度
                # decoder是四个模块的列表 分别输入c3 c2 c1 c0  decoder 还进行了特征的融合
                # decoder返回的是 x_clean and x_ref 通过通道拼接在一起 B,6,H,W 是提取的各种特征并重建的图像 一个子网络就重建一福B,6,H,W图像
                # 为什么相减？通过​​残差学习（Residual Learning）​​让模型专注于学习输入图像与解码器重构图像之间的​​差异信息​​，而非直接生成完整图像
                x_img_out.append(torch.cat([x_in, x_in], dim=-3) - self.decoder_blocks[-1](c3, c2, c1, c0) )
                # decoder_blocks[-1] 是尽管level变化 都固定使用最后一个解码器模块，但最后一个解码器模块与其他模块其实是一样的……

                # subnet0的输出是 x_img_out[0]  subnet1的输出是 x_img_out[1]  subnet2的输出是 x_img_out[2]  subnet3的输出是 x_img_out[3]
        return x_cls_out, x_img_out # x_cls_out什么也没操作 是空的  x_img_out 是4长度的 B,6,H,W的列表

    # 对卷积和全连接层使用截断正态分布初始化（trunc_normal_），偏置项初始化为0，确保训练稳定性
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)

   



# FullNet_NLP
# ├── Input: x_in (B,3,H,W)
# │
# ├── Stem Processing
# │   ├── stem_comp: 5x5 Conv (3->32) + LayerNorm
# │   └── baseball: FocalNet提取多尺度特征 → (c0,c1,c2,c3)
# │       │── baseball_adapter: 1x1 Conv调整通道数
# │       │   ├── c0: 192→64
# │       │   ├── c1: 192→128
# │       │   ├── c2: 384→256 
# │       │   └── c3: 768→512
# │       └── prompt机制（条件α融合）
# │           ├── prompt网络: Linear(6→512) → StarReLU → Linear(512→32)
# │           └── x_stem = prompt_alpha * x_stem (通道调制)
# │
# ├── SubNet 子网络处理 (循环num_subnet次)  子网络就是 把focalnet学到的各层次特征 在subnet仿射变换 在level的时候上下层交流融合
# │   │── SubNet0 (first_col=True)
# │   │   ├── Level0
# │   │   │   ├── Fusion0: 纯下采样(c0→c1)
# │   │   │   └── 2个 ConvNextBlock (带残差)
# │   │   └── 层级连接公式：
# │   │       c0 = α0*c0 + level0(x_stem, c1)
# │   │
# │   ├── SubNet1
# │   │   ├── Level1
# │   │   │   ├── Fusion1: 上采样(c2) + 下采样(c1→c2)
# │   │   │   └── 3x ConvNextBlock
# │   │   └── c1 = α1*c1 + level1(c0, c2)
# │   │
# │   ├── SubNet2
# │   │   ├── Level2
# │   │   │   ├── Fusion2: 上采样(c3) + 下采样(c2→c3) 
# │   │   │   └── 6x ConvNextBlock
# │   │   └── c2 = α2*c2 + level2(c1, c3)
# │   │
# │   └── SubNet3
# │       ├── Level3
# │       │   ├── Fusion3: 纯下采样(c3)
# │       │   └── 3x ConvNextBlock
# │       └── c3 = α3*c3 + level3(c2, None)
# │
# ├── Decoder 解码重构
# │   ├── Decoder Block[-1] (最终解码器)
# │   │   ├── 特征融合路径：
# │   │   │   c3 → Upsample → *c2 → Upsample → *c1 → Upsample → *c0
# │   │   └── 重构流程：
# │   │       ┌───────────────┐
# │   │       │ 双线性上采样  │
# │   │       │ 通道注意力SCA │
# │   │       │ NAFBlock处理  │
# │   │       └───────────────┘
# │   └── 残差计算：
# │       x_img_out.append(cat(x_in,x_in) - decoder_output)
# │
# └── Outputs
#     ├── x_cls_out: []（未实现分类输出）
#     └── x_img_out: [残差图像1, 残差图像2...] (B,6,H,W)
# 关键数据流：

# 输入图像 → Stem预处理 → 多尺度特征提取 → 条件提示调制
#           ↓
# 子网络0 → 子网络1 → 子网络2 → 子网络3 （层级递进处理）
#           │          │          │          │
#           c0         c1         c2         c3 （多尺度特征更新）
#           ↓          ↓          ↓          ↓
# 解码器 ← 特征融合 ← 特征融合 ← 特征融合 ← 最终特征
#           │
#           └─→ 残差输出 = [双倍输入通道] - 解码重构结果 ：四元素
