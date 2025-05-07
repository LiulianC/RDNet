import os
import torch
import util.util as util
import torch.nn as nn


# BaseModel 类是一个 ​​深度学习模型的基础抽象类​​，通常用于 ​​生成对抗网络（GAN）​​ 或 ​​其他复杂模型训练框架​​（如 PyTorch 项目）
class BaseModel(nn.Module):
    def name(self):
        return self.__class__.__name__.lower()

    def initialize(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        last_split = opt.checkpoints_dir.split('/')[-1]
        if opt.resume and last_split != 'checkpoints' and (last_split != opt.name or opt.supp_eval):

            self.save_dir = opt.checkpoints_dir
            self.model_save_dir = os.path.join(opt.checkpoints_dir.replace(opt.checkpoints_dir.split('/')[-1], ''),opt.name)
        else:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.model_save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self._count = 0

    def set_input(self, input):
        self.input = input

    def forward(self, mode='train'):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass # 这是一个空方法（只有 pass）

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def print_optimizer_param(self):
        print(self.optimizers[-1])



    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)



