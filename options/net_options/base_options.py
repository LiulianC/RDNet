from options.base_option import BaseOptions as Base
from util import util
import os
import torch
import numpy as np
import random

class BaseOptions(Base):
    def initialize(self):
        Base.initialize(self)
        # experiment specifics
        self.parser.add_argument('--inet', type=str, default='ytmt_ucs', help='chooses which architecture to use for inet.')
        
        # self.parser.add_argument('--icnn_path', type=str, default='None', help='icnn checkpoint to use.')
        self.parser.add_argument('--icnn_path', type=str, default='D:\gzm-RDNet\RDNet\checkpoints\ytmt_ucs_sirs\ytmt_ucs_sirs_latest.pth', help='icnn checkpoint to use.')
        
        self.parser.add_argument('--init_type', type=str, default='edsr', help='network initialization [normal|xavier|kaiming|orthogonal|uniform]')
        # for network
        self.parser.add_argument('--hyper', action='store_true', help='if true, augment input with vgg hypercolumn feature')
        

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        # self.opt 是一个命名空间对象（argparse.Namespace），包含所有已定义的参数及其值
        # self.parser 是一个 argparse.ArgumentParser 对象，定义了所有可接受的命令行参数（如 --batch_size, --lr 等）
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        # 为什么要设置随机种子？
        # 相同的随机种子会生成相同的随机数序列，确保以下操作一致：
        # 模型权重初始化、数据增强（如随机裁剪、翻转）、数据加载顺序
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed) # seed for every module
        random.seed(self.opt.seed)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0]) # 设置当前使用的 GPU 设备

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        if self.opt.debug:
            self.opt.display_freq = 20
            self.opt.print_freq = 20
            self.opt.nEpochs = 40
            self.opt.max_dataset_size = 100
            self.opt.no_log = False
            self.opt.nThreads = 0
            self.opt.decay_iter = 0
            self.opt.serial_batches = True
            self.opt.no_flip = True
        
        return self.opt
