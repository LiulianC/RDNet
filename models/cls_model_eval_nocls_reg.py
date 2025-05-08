import torch
from torch import nn
import torch.nn.functional as F
from models.losses import DINOLoss
import os
import numpy as np
from collections import OrderedDict
# from ema_pytorch import EMA
from models.arch.classifier import PretrainedConvNext
import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch
#from models.arch.dncnn import effnetv2_s
from .base_model import BaseModel
from PIL import Image
from os.path import join
#from torchviz import make_dot
from models.arch.RDnet_ import FullNet_NLP
import timm

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


# 这个 EdgeMap 类是一个 ​​基于梯度的边缘检测模块​​，继承自 PyTorch 的 nn.Module。它的核心功能是通过计算图像在水平和垂直方向的梯度幅值，生成边缘响应图（edge map）。
class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs().sum(dim=1, keepdim=True)
        # img[..., 1:, :]​​选取所有维度（...），但在 ​​高度（H）维度​​ 上从第 2 个像素开始（1:），保留所有水平通道（:）。相当于去掉第一行的像素
        # 在高度维度上取到倒数第 2 个像素（:-1），保留所有通道。相当于原始图像去掉最后一行的像素（计算左侧像素的差值）。
        grady = (img[..., 1:] - img[..., :-1]).abs().sum(dim=1, keepdim=True)

        gradX[..., :-1, :] += gradx # 选中上半部分
        gradX[..., 1:, :] += gradx # 选中下半部分
        gradX[..., 1:-1, :] /= 2 # 中间部分 相加取均值 还原成图片的size

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge



# ​​显式操作​​：边缘提取（edge_map）是可见的预处理步骤。
# ​​隐式核心操作​​：神经网络模型对输入图像进行深度处理（如去雾、反射分离），隐藏在 forward() 方法中。
# ​​代码结构提示​​：类名 YTMTNetBase 中的 "Net" 表明这是一个神经网络基类，需配合具体网络架构实现图像处理。
class YTMTNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        identity = False
        mode = mode.lower() # 返回原字符串的全小写版本（非字母字符不变）
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])

        self.input = input
        self.identity = identity
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

        self.issyn = False if 'real' in data else True 
        self.aligned = False if 'unaligned' in data else True

        if target_t is not None:
            self.target_edge = self.edge_map(self.target_t)  # 这算出来后没有使用

    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        self._eval()
        self.set_input(data, 'eval')
        with torch.no_grad():
            self.forward_eval() # 虽然没有要返回值 但是 self.output_j 还是会被赋值的

            output_i = tensor2im(self.output_j[6]) # clean
            output_j = tensor2im(self.output_j[7]) # reflection
            target = tensor2im(self.target_t)
            target_r = tensor2im(self.target_r)

            if self.aligned:
                res = index.quality_assess(output_i, target) # 送入计算量化指标
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    savedir = join(savedir, suffix, name)
                    os.makedirs(savedir, exist_ok=True)
                    Image.fromarray(output_i.astype(np.uint8)).save(
                        join(savedir, '{}_t.png'.format(self.opt.name)))
                    Image.fromarray(output_j.astype(np.uint8)).save(
                        join(savedir, '{}_r.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(
                        join(savedir, 'transmission_layer', str(self._count) + '.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(
                        join(savedir, 'blended', str(self._count) + '.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return

        with torch.no_grad():
            output_i, output_j = self.forward()
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            if self.data_name is not None and savedir is not None:
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}_l.png'.format(self.opt.name)))
                Image.fromarray(output_j.astype(np.uint8)).save(join(savedir, name, '{}_r.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))




# ClsModel 类是一个 ​​结合多任务学习与对抗训练的复杂图像分解模型​​，专门用于 ​​反射层与透射层的分离​​（如去玻璃反光、去雾）或 ​​图像增强任务​​。
# 它继承自 YTMTNetBase，核心设计融合了 ​​预训练特征提取​​、​​生成对抗网络（GAN）​​ 和 ​​多级损失约束​​。
class ClsModel(YTMTNetBase):
    def name(self):
        return 'ytmtnet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net_c = None

    def print_networks(self): # 网络结构打印​
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval() # net_i​​：自定义的 FullNet_NLP 网络，负责生成 ​​透射层（主体内容）​​ 和 ​​反射层（干扰成分）​​。
        self.net_c.eval() # net_c​​：预训练的 ConvNeXt 模型（PretrainedConvNext），用于提取高层语义特征。

    def _train(self):
        self.net_i.train() # 生成器设为训练模式（启用Dropout/BatchNorm）
        self.net_c.eval()  # 特征提取网络设为评估模式（固定参数）

    def initialize(self, opt):# 核心初始化​
        self.opt=opt
        BaseModel.initialize(self, opt)

        in_channels = 3
        self.vgg = None

        if opt.hyper:
            self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
            in_channels += 1472 # # VGG多层级特征拼接后的通道数

        channels = [64, 128, 256, 512]  # 各层通道数 
        layers = [2, 2, 4, 2]           # 各层块数
        num_subnet = opt.num_subnet     # 子网络数量（来自配置）

        # 初始化预训练特征提取网络（ConvNeXt）
        self.net_c = PretrainedConvNext("convnext_small_in22k").cuda()
        self.net_c.load_state_dict(torch.load('D:/gzm-RDNet/RDNet/models/pretrained/cls_model.pth')['icnn'])

        # 初始化主生成器网络（FullNet_NLP）
        self.net_i = FullNet_NLP(channels, layers, num_subnet, opt.loss_col, num_classes=1000, drop_path=0,save_memory=True, inter_supv=True, head_init_scale=None, kernel_size=3).to(self.device)
    
        # 边缘检测模块（用于辅助损失）
        self.edge_map = EdgeMap(scale=1).to(self.device)
    
        # 训练相关初始化
        # 初始化一个字典 键是字符串 值是loss函数 vgg是后面添加的
        if self.isTrain:
            self.loss_dic = losses.init_loss(opt, self.Tensor)  # 初始化损失函数字典
            
            # 添加VGG感知损失
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['t_vgg'] = vggloss

            # 配置内容损失（多种类型可选）
            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.vgg, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.vgg, weights=[0.1, 0.1, 0.1, 0.1], indices=[8, 13, 22, 31],
                                                criterions=[losses.CX_loss] * 3 + [nn.L1Loss()]))
            else:
                raise NotImplementedError
            self.loss_dic['t_cx'] = cxloss
            
            # 混合精度训练配置
            # torch.cuda.amp.GradScaler()创建了一个梯度缩放器实例。这个实例负责在训练过程中根据梯度的动态范围，自动调整放缩系数
            self.scaler=torch.cuda.amp.GradScaler()
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                self.dinoloss=DINOLoss() # 自监督对比损失

            # 生成器优化器
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
            self._init_optimizer([self.optimizer_G])

        # 恢复训练检查点
        if opt.resume:
            self.load(self, opt.resume_epoch)

        if opt.print_networks is not None:
            if opt.print_networks:
                self.print_networks()


    def load_networks(self):
        if self.opt.model_path is not None:
            print('\n')
            print('--------------------------------------------')
            print('Load the model from %s ' % self.opt.icnn_path)
            icnn_path = self.opt.icnn_path
            state_dict = torch.load(icnn_path)
            self.net_i.load_state_dict(state_dict['icnn'])
            self.optimizer_G.load_state_dict(state_dict['opt_g'])
            self.epoch = state_dict['epoch']
        else:
            print('No model to load. ')
            pass        
        



    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0: # 如果启动gan网络 算gan损失
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })
        return state_dict

    def state_dict_eval(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }
        return state_dict


    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_%03d_%08d.pth' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_' + label + '.pth')

        torch.save(self.state_dict(), model_name)

    
    def save_eval(self, label=None):
        model_name = os.path.join(self.model_save_dir, label + '.pth')
        torch.save(self.state_dict_eval(), model_name)



    # 判别器反向传播​
    def backward_D(self):
        loss_D=[]
        weight=self.opt.weight_loss     # 判别器反向传播​

        # 启用判别器梯度
        for p in self.netD.parameters():
            p.requires_grad = True

        # 多尺度判别损失计算（4个尺度）    
        for i in range(4):
            loss_D_1, pred_fake_1, pred_real_1 = self.loss_dic['gan'].get_loss(
                self.netD, self.input, self.output_j[2*i], self.target_t)
            loss_D.append(loss_D_1*weight)
            weight+=self.opt.weight_loss
        
        # 合并损失并反向传播
        loss_sum=sum(loss_D)
        self.loss_D, self.pred_fake, self.pred_real = (loss_sum, pred_fake_1, pred_real_1)
        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)


    # 生成器损失计算​
    def get_loss(self, out_l, out_r):
        
        # backward_G 给的是 out_l=output_i out_r=output_j 是元素为8 的的列表 前四元素为(B,6,H,W)后四元素为(B,3,H,W) 目前 output_i是空的

        # 初始化各损失累加器
        loss_G_GAN_sum=[]
        loss_icnn_pixel_sum=[]
        loss_rcnn_pixel_sum=[]
        loss_icnn_vgg_sum=[]
        weight=self.opt.weight_loss

        # 遍历多级输出（loss_col控制级数）
        for i in range(self.opt.loss_col):# i=0 1 2 3

            # 获取当前级输出（clean和reflection）
            # i=0时 out_r_clean=output_j[0] out_r_reflection=output_j[1]
            # i=1时 out_r_clean=output_j[2] out_r_reflection=output_j[3]
            # i=2时 out_r_clean=output_j[4] out_r_reflection=output_j[5]
            # i=3时 out_r_clean=output_j[6] out_r_reflection=output_j[7]
            # output_j 是元素为8 的的列表 前四元素为(B,6,H,W)后四元素为(B,3,H,W)
            out_r_clean=out_r[2*i]
            out_r_reflection=out_r[2*i+1]

            # 非最终级的损失计算
            if i != self.opt.loss_col -1: # 如果 i != 3 
                loss_G_GAN = 0 # 中间层不计算GAN损失
                # 像素级L1损失 # loss_dic是一个字典 键是字符串 值是loss函数
                loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_r_clean, self.target_t)
                # 反射层像素损失（加权）
                loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r_reflection, self.target_r) * 1.5 * self.opt.r_pixel_weight
                 # VGG感知损失
                loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_r_clean, self.target_t) * self.opt.lambda_vgg
            else:
                if self.opt.lambda_gan > 0:

                    loss_G_GAN=0
                else:
                    loss_G_GAN=0
                loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(out_r_clean, self.target_t)
                loss_rcnn_pixel = self.loss_dic['r_pixel'].get_loss(out_r_reflection, self.target_r) * 1.5 * self.opt.r_pixel_weight
                loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(out_r_clean, self.target_t) * self.opt.lambda_vgg

            loss_G_GAN_sum.append(loss_G_GAN*weight)
            loss_icnn_pixel_sum.append(loss_icnn_pixel*weight)
            loss_rcnn_pixel_sum.append(loss_rcnn_pixel*weight)
            loss_icnn_vgg_sum.append(loss_icnn_vgg*weight)
            weight=weight+self.opt.weight_loss
        return sum(loss_G_GAN_sum), sum(loss_icnn_pixel_sum), sum(loss_rcnn_pixel_sum), sum(loss_icnn_vgg_sum)

    
    # 生成器反向传播​ 从 optimize_parameters 而来
    # 计算生成器损失（对抗损失、像素损失、VGG感知损失）
    def backward_G(self):
        # output_j 是4长度的 元素是 B,6,H,W 的列表
        self.loss_G_GAN,self.loss_icnn_pixel, self.loss_rcnn_pixel, self.loss_icnn_vgg = self.get_loss(self.output_i, self.output_j)

        self.loss_exclu = self.exclusion_loss(self.output_i, self.output_j, 3)

        self.loss_recons = self.loss_dic['recons'](self.output_i, self.output_j, self.input) * 0.2

        self.loss_G =  self.loss_G_GAN +self.loss_icnn_pixel + self.loss_rcnn_pixel + self.loss_icnn_vgg
        
        # 将损失值 self.loss_G 乘以当前梯度缩放器（Gradient Scaler）用于动态调整梯度的缩放因子 
        self.scaler.scale(self.loss_G).backward() 


    #  超列特征生成​    # 通过VGG网络提取多层级特征，并进行上采样以匹配输入图像的大小
    def hyper_column(self, input_img):
        hypercolumn = self.vgg(input_img)
        _, C, H, W = input_img.shape
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                       feature in hypercolumn]
        input_i = [input_img]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
        return input_i

    # 前向传播
    def forward(self):
        # without edge
        
        self.output_j=[] 
        input_i = self.input  # 输入数据（混合图像）

        # 可选：生成超列特征
        if self.vgg is not None:
            input_i = self.hyper_column(input_i)

        # 用net_c提取特征（不计算梯度）
        with torch.no_grad():
            ipt = self.net_c(input_i)

        # 主网络生成输出
        # output_i是RDnet的 x_cls_out ,output_j是RDnet的x_img_out  x_cls_out 此时还是空的 x_img_out 是原始图像 减去 提取到的特征又被重建的图像（残差）
        # output_j是4长度的 B,6,H,W的列表
        # subnet0的输出是 output_j[0]  subnet1的输出是 output_j[1]  subnet2的输出是 output_j[2]  subnet3的输出是 output_j[3]
        output_i, output_j = self.net_i(input_i,ipt,prompt=True) 
        self.output_i = output_i

        # 整理多级输出（clean和reflection交替存放）
        for i in range(self.opt.loss_col): # i=0 1 2 3
            # 预测的反射图像是out_reflection=output_j前三通道  预测的投射图像时out_clean=output_j后三通道
            out_reflection, out_clean = output_j[i][:, :3, ...], output_j[i][:, 3:, ...]
            
            # 这里 self.output_j 不是output_j, self.output_j 是长度为8的B,3,H,W的列表

            # i=0 out_reflection,out_clean = output_j[0][:, :3, ...], output_j[0][:, 3:, ...]   抽1元素的output_j 分成两个B,3,H,W 拼到列表 来自子网络0
            # i=1 out_reflection,out_clean = output_j[1][:, :3, ...], output_j[1][:, 3:, ...]   抽1元素的output_j 分成两个B,3,H,W 拼到列表 来自子网络1
            # i=2 out_reflection,out_clean = output_j[2][:, :3, ...], output_j[2][:, 3:, ...]   抽1元素的output_j 分成两个B,3,H,W 拼到列表 来自子网络2
            # i=3 out_reflection,out_clean = output_j[3][:, :3, ...], output_j[3][:, 3:, ...]   抽1元素的output_j 分成两个B,3,H,W 拼到列表 来自子网络3
                                    
            self.output_j.append(out_clean)   # 干净图像层 双数
            self.output_j.append(out_reflection)   # 反射层 单数

        
        return self.output_i, self.output_j




    @torch.no_grad() 
    def forward_eval(self):
       
        self.output_j=[]
        input_i = self.input

        if self.vgg is not None:
            input_i = self.hyper_column(input_i)

        ipt = self.net_c(input_i)
        
        output_i, output_j = self.net_i(input_i,ipt,prompt=True)
        self.output_i = output_i #alpha * output_i + beta

        for i in range(self.opt.loss_col):
            out_reflection, out_clean = output_j[i][:, :3, ...], output_j[i][:, 3:, ...]
            self.output_j.append(out_clean) 
            self.output_j.append(out_reflection)
        return self.output_i, self.output_j


    def optimize_parameters(self):
        self._train()
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def return_output(self):
        output_clean = self.output_j[6]
        output_reflection = self.output_j[7]
        return output_clean, output_reflection,self.input,self.target_r,self.target_t # 加个对照
    

    # 排斥损失（关键创新点）​
    def exclusion_loss(self, img_T, img_R, level=3, eps=1e-6):
        loss_gra=[]
        weight=0.25

        # # 多层级梯度互斥计算
        for i in range(4):
            grad_x_loss = []
            grad_y_loss = []
            img_T=self.output_j[2*i]
            img_R=self.output_j[2*i+1]

            # 下采样循环（level控制）
            for l in range(level):

                # 计算梯度
                grad_x_T, grad_y_T = self.compute_grad(img_T)
                grad_x_R, grad_y_R = self.compute_grad(img_R)

                # 自适应权重平衡
                alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + eps)
                alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + eps)

                 # 梯度归一化（sigmoid转tanh）
                gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
                grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
                gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
                grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

                # 互斥损失计算
                grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + eps) ** 0.25)
                grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + eps) ** 0.25)

                 # 下采样准备下一轮
                img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
                img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')

            # 层级损失加权
            loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)
            loss_gra.append(loss_gradxy*weight)
            weight+=0.25


        return sum(loss_gra) / 2



    def contain_loss(self, img_T, img_R, img_I, eps=1e-6):
        pix_num = np.prod(img_I.shape)
        predict_tx, predict_ty = self.compute_grad(img_T)
        predict_tx, predict_ty = self.compute_grad(img_T)
        predict_rx, predict_ry = self.compute_grad(img_R)
        input_x, input_y = self.compute_grad(img_I)

        out = torch.norm(predict_tx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ty / (input_y + eps), 2) ** 2 + \
              torch.norm(predict_rx / (input_x + eps), 2) ** 2 + \
              torch.norm(predict_ry / (input_y + eps), 2) ** 2

        return out / pix_num

    def compute_grad(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


    
    def get_current_loss(self):
        return self.loss_G, self.loss_icnn_pixel, self.loss_rcnn_pixel, self.loss_icnn_vgg, self.loss_exclu, self.loss_recons # loss都是数 不是列表
    


# AvgPool2d 类是一个 ​​自定义的自适应平均池化层​​，设计用于在 ​​动态输入尺寸​​ 场景下高效工作
# ​面向动态分辨率场景优化​​ 的自适应池化层，通过动态核调整+积分图技巧，在 ​​大核池化​​ 和 ​​多尺度应用​​ 中展现出比标准实现更高的效率。常用于计算机视觉中对分辨率敏感的模型（如 GAN、超分辨率）
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):

        # 动态计算池化核 (forward)​
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)
