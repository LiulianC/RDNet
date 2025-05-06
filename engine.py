import torch
import util.util as util
from models import make_model
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer
from tqdm import tqdm
import visdom
import numpy as np
from tools import mutils
import torch
from models.cls_model_eval_nocls_reg import ClsModel

class Engine(object):
    def __init__(self, opt,eval_dataset_real,eval_dataset_solidobject,eval_dataset_postcard,eval_dataloader_wild):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6
        self.eval_dataset_real = eval_dataset_real
        self.eval_dataset_solidobject = eval_dataset_solidobject
        self.eval_dataset_postcard = eval_dataset_postcard
        self.eval_dataloader_wild = eval_dataloader_wild
        self.result_dir = os.path.join(f'./experiment/{self.opt.name}/results',
                          mutils.get_formatted_time())
        self.biggest_psnr=0
        self.__setup()

    def __setup(self):
        self.basedir = join('experiment', self.opt.name)
        os.makedirs(self.basedir, exist_ok=True)

        opt = self.opt

        """Model"""
        self.model = make_model(self.opt.model)  # models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        if True:
            print("IN")
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)




    def train(self, train_loader, **kwargs): # **kwargs 是 Python 函数定义中的一种参数写法，意思是“关键字参数字典”。
        """Train"""                         # key words array
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters() # 初始化损失统计器
        opt = self.opt
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()

        train_pbar = tqdm(train_loader)

        for i, data in enumerate(train_pbar):

            iter_start_time = time.time()
            iterations = self.iterations

            # 模型前向传播与优化
            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs) # ​​完成一次生成器（net_i）的参数更新

            errors = model.get_current_errors() # 返回一个空矩阵
            avg_meters.update(errors)           #
            # util.progress_bar(i, len(train_loader), str(avg_meters)) # 进度条

            

            util.write_loss(self.writer, 'train', avg_meters, iterations)  # 记录loss
            

            # 定期可视化（每100次迭代）
            if iterations % 100 ==0:
                imgs=[]
                output_clean,output_reflection,input=model.return_output() # 从模型 得到 T R I
                # print('output_clean size: ',(output_clean.shape)) # 是三通道的！
                
                # 数据格式转换（HWC -> CHW）并归一化
                output_clean=np.transpose(output_clean,(2,0,1))/255
                #output_reflection = np.transpose(output_reflection, (2, 0, 1))/255
                input = np.transpose(input, (2, 0, 1))/255
                imgs.append(output_clean)
                #imgs.append(output_reflection)
                imgs.append(input)
                util.get_visual(self.writer,iterations,imgs) # 将图像写入TensorBoard
                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)

            self.iterations += 1  # 更新迭代次数

            loss_G,loss_icnn_pixel,loss_rcnn_pixel,loss_icnn_vgg,loss_exclu,loss_recons=model.get_current_loss()
            train_pbar.update(1)
            train_pbar.set_postfix({'loss': loss_G.item()})            

        self.epoch += 1

        if True:#not self.opt.no_log:
            if self.epoch % opt.save_epoch_freq == 0:
                save_dir = os.path.join(self.result_dir, '%03d' % self.epoch)
                os.makedirs(save_dir, exist_ok=True)
                matrix_real=self.eval(self.eval_dataset_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')
                matrix_solid=self.eval(self.eval_dataset_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,
                    suffix='solidobject')
                matrix_post=self.eval(self.eval_dataset_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')
                matrix_wild=self.eval(self.eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')
                sum_PSNR_real=matrix_real['PSNR']*20
                sum_PSNR_solid=matrix_solid['PSNR']*200
                sum_PSNR_post=matrix_post['PSNR']*199
                sum_PSNR_wild=matrix_wild['PSNR']*55
                print("sum_PSNR_real: ",matrix_real['PSNR'],"sum_PSNR_solid: ",matrix_solid['PSNR'],"sum_PSNR_post: ",matrix_post['PSNR'],"sum_PSNR_wild: ",matrix_wild['PSNR'])
                sum_PSNR = float(sum_PSNR_real + sum_PSNR_solid + sum_PSNR_post + sum_PSNR_wild)/474.0
                print('总PSNR:', sum_PSNR)
                if sum_PSNR>self.biggest_psnr:
                    self.biggest_psnr=sum_PSNR
                    print('saving the model at epoch %d, iters %d' %(self.epoch, self.iterations))
                    model.save()
                print('highest： ',self.biggest_psnr,' name: ',opt.name)

            print('saving the latest model at the end of epoch %d, iters %d' %
                  (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                  (time.time() - epoch_start_time))

        # model.update_learning_rate()
        try:
            train_loader.reset()
        except:
            pass

        train_pbar.close()


    def eval(self, val_loader, dataset_name, savedir='./tmp', loss_key=None, **kwargs):
        # print(dataset_name)
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            self.f = open(os.path.join(savedir, 'metrics.txt'), 'w+')
            self.f.write(dataset_name + '\n')
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if self.opt.select is not None and data['fn'][0] not in [f'{self.opt.select}.jpg']:
                    continue
                #print(data.shape())
                index = model.eval(data, savedir=savedir, **kwargs)

                # print(data['fn'][0], index)
                if savedir is not None:
                    self.f.write(f"{data['fn'][0]} {index['PSNR']} {index['SSIM']}\n")
                avg_meters.update(index)
                util.progress_bar(i, len(val_loader), str(avg_meters))

        if not opt.no_log:
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)

        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' %
                      (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    def save_eval(self, label):
        self.model.save_eval(label)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
