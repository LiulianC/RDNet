import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_msssim import SSIM
from models.vit_feature_extractor import VitExtractor
from models.vgg import Vgg19
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torchvision import models


###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class ContainLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(ContainLoss, self).__init__()
        self.eps = eps

    def forward(self, predict_t, predict_r, input_image):
        pix_num = np.prod(input_image.shape)
        predict_tx, predict_ty = compute_gradient(predict_t)
        predict_rx, predict_ry = compute_gradient(predict_r)
        input_x, input_y = compute_gradient(input_image)

        out = torch.norm(predict_tx / (input_x + self.eps), 2) ** 2 + \
              torch.norm(predict_ty / (input_y + self.eps), 2) ** 2 + \
              torch.norm(predict_rx / (input_x + self.eps), 2) ** 2 + \
              torch.norm(predict_ry / (input_y + self.eps), 2) ** 2

        return out / pix_num


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d): 
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        # torch.eye(c) 输出​​：一个 c × c 的二维张量（矩阵），其 ​​主对角线元素为 1​​，其余元素为 0。
        self.weight.data = torch.eye(c).view(c, c, 1, 1) # 将单位矩阵转换为 ​​1x1 卷积核的权重​​。(out_channels, in_channels, kernel_height, kernel_width)
        # 为什么(out_channels, in_channels, kernel_height, kernel_width) 是 卷积核的size 真奇怪，现存疑
        if norm:
            # .data 的作用​ 返回一个与原始张量共享存储空间但 ​​剥离计算图​​ 的新张量 不跟踪梯度​​ 直接修改原始数据​
            # .div_() 原地除法操作​​（in-place division），等效于 x = x / y，但直接修改原张量。
            # 原地权重调整: weight = 1 / std
            self.weight.data.div_(std.view(c, 1, 1, 1))
            # 偏置调整: bias = (-mean * data_range) / std
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False # 归一化参数不更新

        # 为什么没有return却能return？因为继承自2d卷积 所以会自动return 


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            # self.vgg = torch.compile(Vgg19().cuda())
            self.vgg = models.vgg19(pretrained=True).features.to('cuda')
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        # 经验性设置，深层（如第五层）权重较大，强调语义对齐。
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        # 浅层（如 relu1_2）捕捉边缘、颜色等低级特征。 深层（如 relu5_2）捕捉物体结构、语义等高级特征。
        self.indices = indices or [2, 7, 12, 21, 30] # relu1_2、relu2_2、relu3_2、relu4_2、relu5_2
        
        if normalize:
            # 第一个数组是平均值的意思 第二个数组是标准差
            # 为什么给的是三元素数组？
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        # vgg只是一个固定网络 用来提取x与y的信息 为什么算x时要保存梯度 算y时不要？
        # 即使 VGG 的参数是固定的，​​计算图中仍会保留 x 到 loss 的路径​​，从而允许梯度传递到图像生成器（消除反光）。
        # loss → L1梯度 → x_vgg梯度 → VGG网络的反向计算 → x的梯度 → G的参数梯度
        loss = []
        for i, layer in enumerate(self.vgg): # 遍历 VGG19 模型的每一层
            x = layer(x) # 将输入图像 x 通过当前层。
            with torch.no_grad(): 
                y = layer(y) # 将目标图像 y 通过当前层
            if i in self.indices: 
                loss.append(F.l1_loss(x, y)) 

        for i in range(len(loss)):
            loss[i] = self.weights[i] * loss[i]
        vgg_loss = sum(loss)
        return vgg_loss
    
    def compute_perceptual_loss(x, y):
        loss = 0.0
        for i, layer in enumerate(vgg): # 遍历 VGG19 模型的每一层
            x = layer(x) # 将输入图像 x 通过当前层。
            y = layer(y) # 将目标图像 y 通过当前层
            if i in {1, 6, 11, 20, 29}: # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 # 这些索引对应 VGG19 模型中的特定卷积层（conv1_2、conv2_2、conv3_3、conv4_3、conv5_3）。
                loss += F.l1_loss(x, y) # 如果当前层是这些特定卷积层之一，则计算 x 和 y 之间的 L1 损失，并将其加到 loss 中。
        return loss


def l1_norm_dim(x, dim):
    return torch.mean(torch.abs(x), dim=dim)


def l1_norm(x):
    return torch.mean(torch.abs(x))


def l2_norm(x):
    return torch.mean(torch.square(x))


def gradient_norm_kernel(x, kernel_size=10):
    out_h, out_v = compute_gradient(x)
    shape = out_h.shape
    out_h = F.unfold(out_h, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_h = out_h.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_h = l1_norm_dim(out_h, 2)
    out_v = F.unfold(out_v, kernel_size=(kernel_size, kernel_size), stride=(1, 1))
    out_v = out_v.reshape(shape[0], shape[1], kernel_size * kernel_size, -1)
    out_v = l1_norm_dim(out_v, 2)
    return out_h, out_v


class KTVLoss(nn.Module):
    def __init__(self, kernel_size=10):
        super().__init__()
        self.kernel_size = kernel_size
        self.criterion = nn.L1Loss()
        self.eps = 1e-6

    def forward(self, out_l, out_r, input_i):
        out_l_normx, out_l_normy = gradient_norm_kernel(out_l, self.kernel_size)
        out_r_normx, out_r_normy = gradient_norm_kernel(out_r, self.kernel_size)
        input_normx, input_normy = gradient_norm_kernel(input_i, self.kernel_size)
        norm_l = out_l_normx + out_l_normy
        norm_r = out_r_normx + out_r_normy
        norm_target = input_normx + input_normy + self.eps
        norm_loss = (norm_l / norm_target + norm_r / norm_target).mean()

        out_lx, out_ly = compute_gradient(out_l)
        out_rx, out_ry = compute_gradient(out_r)
        input_x, input_y = compute_gradient(input_i)
        gradient_diffx = self.criterion(out_lx + out_rx, input_x)
        gradient_diffy = self.criterion(out_ly + out_ry, input_y)
        grad_loss = gradient_diffx + gradient_diffy

        loss = norm_loss * 1e-4 + grad_loss
        return loss


class MTVLoss(nn.Module):
    def __init__(self, kernel_size=10):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.norm = l1_norm

    def forward(self, out_l, out_r, input_i):
        out_lx, out_ly = compute_gradient(out_l)
        out_rx, out_ry = compute_gradient(out_r)
        input_x, input_y = compute_gradient(input_i)

        norm_l = self.norm(out_lx) + self.norm(out_ly)
        norm_r = self.norm(out_rx) + self.norm(out_ry)
        norm_target = self.norm(input_x) + self.norm(input_y)

        gradient_diffx = self.criterion(out_lx + out_rx, input_x)
        gradient_diffy = self.criterion(out_ly + out_ry, input_y)

        loss = (norm_l / norm_target + norm_r / norm_target) * 1e-5 + gradient_diffx + gradient_diffy

        return loss


class ReconsLoss(nn.Module):
    def __init__(self, edge_recons=True):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.norm = l1_norm
        self.edge_recons = edge_recons
        self.mse_loss=nn.MSELoss()

    def forward(self, out_l, out_r, input_i):
        loss_sum=[]
        weight=0.25
        for i in range(4):
            #out_res = out_l[i]
            out_clean=out_r[2*i]
            out_reflection=out_r[2*i+1]
            #content_diff = self.criterion(out_clean + out_reflection, input_i)
            # if self.edge_recons:
            #     out_lx, out_ly = compute_gradient(out_clean)
            #     out_rx, out_ry = compute_gradient(out_reflection)
            #     #out_resx, out_resy = compute_gradient(out_res)
            #     input_x, input_y = compute_gradient(input_i)

            #     gradient_diffx = self.criterion(out_lx + out_rx, input_x)
            #     gradient_diffy = self.criterion(out_ly + out_ry, input_y)

            #     loss = content_diff + (gradient_diffx + gradient_diffy) * 5.0
            # else:
            #     loss = content_diff
            loss=self.mse_loss(out_clean+out_reflection,input_i)
            loss_sum.append(loss*weight)
            weight=weight+0.25

        return sum(loss_sum)


class ReconsLossX(nn.Module):
    def __init__(self, edge_recons=True):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.norm = l1_norm
        self.edge_recons = edge_recons

    def forward(self, out, input_i):
        content_diff = self.criterion(out, input_i)
        if self.edge_recons:
            out_x, out_y = compute_gradient(out)
            input_x, input_y = compute_gradient(input_i)

            gradient_diffx = self.criterion(out_x, input_x)
            gradient_diffy = self.criterion(out_y, input_y)

            loss = content_diff + (gradient_diffx + gradient_diffy) * 1.0
        else:
            loss = content_diff
        return loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()  # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from 
    # https://arxiv.org/abs/1807.00734        
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1)  # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from 
    # https://arxiv.org/abs/1807.00734    
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)

        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real






class SSIM(nn.Module):
    def __init__(self, window_size=11, channel=3, data_range=1.0, size_average=True):
        """
        Structural Similarity Index (SSIM) 模块
        :param window_size: 高斯窗口大小（必须为奇数）
        :param channel: 输入图像的通道数（1或3）
        :param data_range: 像素值范围（如0-1为1.0，0-255为255.0）
        :param size_average: 是否对空间维度取平均
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.size_average = size_average

        # 生成高斯权重核
        self.gaussian_kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        # 生成1D高斯核
        sigma = 1.5  # 经验值，对应窗口大小11
        coords = torch.arange(self.window_size).float()
        coords -= (self.window_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        # 生成2D高斯核（外积）
        gaussian = torch.outer(g, g)
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # 扩展到多通道
        gaussian = gaussian.repeat(self.channel, 1, 1, 1)  # [C,1,H,W]
        return nn.Parameter(gaussian, requires_grad=False)

    def forward(self, img1, img2):
        """
        计算两个图像的SSIM
        :param img1: 输入图像1 [B,C,H,W]
        :param img2: 输入图像2 [B,C,H,W]
        :return: SSIM值或SSIM图
        """
        # 输入检查
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")
        if self.window_size > min(img1.shape[2], img1.shape[3]):
            raise ValueError("Window size exceeds image dimensions")

        # 数据范围相关常数
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # 应用高斯滤波计算局部统计量
        def gaussian_conv(x):
            return F.conv2d(x, self.gaussian_kernel.to(x.device), 
                           padding=self.window_size//2, groups=self.channel)

        # 计算均值
        mu1 = gaussian_conv(img1)
        mu2 = gaussian_conv(img2)

        # 计算方差和协方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_conv(img1 * img1) - mu1_sq
        sigma2_sq = gaussian_conv(img2 * img2) - mu2_sq
        sigma12 = gaussian_conv(img1 * img2) - mu1_mu2

        # SSIM公式
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map









class SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, output, target):
        return 1 - self.ssim(output, target)


def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    loss_dic = {}

    pixel_loss = ContentLoss()
    pixel_loss.initialize(MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6]))

    loss_dic['t_pixel'] = pixel_loss

    r_loss = ContentLoss()
    r_loss.initialize(MultipleLoss([nn.MSELoss()], [0.9]))
    loss_dic['r_pixel'] = pixel_loss

    loss_dic['t_ssim'] = SSIM_Loss()
    loss_dic['r_ssim'] = SSIM_Loss()

    loss_dic['mtv'] = MTVLoss()
    loss_dic['ktv'] = KTVLoss()
    loss_dic['recons'] = ReconsLoss(edge_recons=False)
    loss_dic['reconsx'] = ReconsLossX(edge_recons=False)

    if opt.lambda_gan > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic

class DINOLoss(nn.Module):
    '''
    DINO-ViT as perceptual loss
    '''

    def resize_to_dino(self, feature, size = (224, 224)): 
        return F.interpolate(feature, size = size, mode='bilinear', align_corners=False)

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0)
            b = self.global_transform(b).unsqueeze(0)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def __init__(self) :
        super(DINOLoss, self).__init__()
        self.extractor = VitExtractor(model_name = 'dino_vits8', device = 'cuda')
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()

    def forward(self, output, target):
        output = self.normalize(self.resize_to_dino(output))
        output_cls_token = self.extractor.get_feature_from_input(output)[-1][0, 0, :]
        with torch.no_grad():
            target = self.normalize(self.resize_to_dino(target))
            target_cls_token = self.extractor.get_feature_from_input(target)[-1][0, 0, :]

        return F.mse_loss(output_cls_token, target_cls_token)
    
if __name__ == '__main__':
    x = torch.randn(3, 32, 224, 224).cuda()
    import time

    s = time.time()
    out1, out2 = gradient_norm_kernel(x)
    t = time.time()
    print(t - s)
    print(out1.shape, out2.shape)
