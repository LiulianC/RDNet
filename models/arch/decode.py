import torch.nn as nn

# 构建 VGG 网络作为特征编码器​​（即图像特征提取器），
# 主要用于计算机视觉任务中的 ​​特征提取​​


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 配置 'E' 对应 VGG19 的经典结构：
# ​​数字​​：卷积层输出通道数（如 64 表示 64 个 3x3 卷积核）
# 'M'​​：插入 2x2 最大池化层（分辨率减半）
cfgs = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

        
class VGG(nn.Module):
    def __init__(self,features):
        super(VGG, self).__init__()
        self.features = features
        
    def forward(self, x):
        x = self.features(x)
        
def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def encoder(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)