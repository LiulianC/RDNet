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
from torchvision.utils import save_image 

class Engine(object):
    def __init__(self, opt, eval_dataloader_gen_scenery, eval_dataloader_gen_tissue, eval_dataloader_real_tissue):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.eval_dataloader_gen_scenery = eval_dataloader_gen_scenery
        self.eval_dataloader_gen_tissue = eval_dataloader_gen_tissue
        self.eval_dataloader_real_tissue = eval_dataloader_real_tissue

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
        self.model.load_networks()
        if True:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)




    def train(self, train_loader, **kwargs): # **kwargs 是 Python 函数定义中的一种参数写法，意思是“关键字参数字典”。
        """Train"""                         # key words array
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters() # 初始化损失统计器
        opt = self.opt
        model = self.model

        epoch_start_time = time.time()


        train_pbar = tqdm(
                train_loader,
                desc="Training",      # 左侧描述文字
                total=len(train_loader),  # 确保总数正确
                ncols=100,               # 固定宽度（根据图中格式建议）
                dynamic_ncols=False,    # 禁用动态宽度
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                # leave=False             # 训练结束后自动清除
            )

        for i, data in enumerate(train_pbar):

            iter_start_time = time.time()
            iterations = self.iterations

            # 模型前向传播与优化
            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs) # ​​完成一次生成器（net_i）的参数更新

            errors = model.get_current_errors() # 返回一个空矩阵
            # util.progress_bar(i, len(train_loader), str(avg_meters)) # 进度条

            

            util.write_loss(self.writer, 'train', avg_meters, iterations)  # 记录loss
            

            # 定期可视化（每100次迭代）
            if iterations % 100==0:
                imgs=[]
                output_clean, output_reflection, input1, _, _ =model.return_output() # 从模型 得到 T R I
                # print('output_clean size: ',(output_clean.shape)) # 是三通道的！
                
                # 数据格式转换（HWC -> CHW）并归一化
                imgs.append(input1)
                imgs.append(output_clean)
                imgs.append(output_reflection)
                util.get_visual(self.writer,iterations,imgs) # 将图像写入TensorBoard
                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)
                

            self.iterations += 1  # 更新迭代次数

            loss_G,loss_icnn_pixel,loss_rcnn_pixel,loss_icnn_vgg,loss_exclu,loss_recons=model.get_current_loss()
            # train_pbar.set_postfix({'loss': loss_G.item()})   
            train_pbar.set_postfix({
                'loss': f'{loss_G.item():.4f}'      # 四舍五入到小数点后四位
                # 'lr': optimizer.param_groups[0]['lr']
            }, refresh=False)  # 手动控制刷新频率                     
            train_pbar.update(1)
            # avg_meters.update(errors)           #

        self.epoch += 1

        if True:#not self.opt.no_log:
            # if self.epoch % opt.save_epoch_freq == 0:
            if self.epoch % 1 == 0:
                save_dir = os.path.join(self.result_dir, '%03d' % self.epoch)
                os.makedirs(save_dir, exist_ok=True)

                matrix_gen_scenery = self.eval(self.eval_dataloader_gen_scenery, dataset_name='eval_dataloader_gen_scenery', savedir=save_dir, suffix='gen_scenery')
                matrix_gen_tissue  = self.eval(self.eval_dataloader_gen_tissue, dataset_name='eval_dataloader_gen_tissue', savedir=save_dir, suffix='gen_tissue')
                matrix_real_tissue = self.eval(self.eval_dataloader_real_tissue, dataset_name='eval_dataloader_real_tissue', savedir=save_dir, suffix='real_tissue')
                sum_PSNR_gen_scenery=matrix_gen_scenery['PSNR']*100
                sum_PSNR_gen_tissue=matrix_gen_tissue['PSNR']*100
                sum_PSNR_real_tissue=matrix_real_tissue['PSNR']*41

                # print("sum_PSNR_gen_scenery: ",sum_PSNR_gen_scenery,"sum_PSNR_gen_tissue: ",sum_PSNR_gen_tissue,"sum_PSNR_real_tissue: ",sum_PSNR_real_tissue)

                sum_PSNR = float(sum_PSNR_gen_scenery + sum_PSNR_gen_tissue + sum_PSNR_real_tissue)/241
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
            val_pbar = tqdm(
                val_loader,
                desc="Validating",
                total=len(val_loader),
                ncols=100,  # 建议宽度根据指标数量调整
                dynamic_ncols=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            for i, data in enumerate(val_pbar):
                if self.opt.select is not None and data['fn'][0] not in [f'{self.opt.select}.jpg']:
                    continue
                #print(data.shape())
                index = model.eval(data, savedir=savedir, **kwargs)

                # print(data['fn'][0], index)
                if savedir is not None:
                    self.f.write(f"{data['fn'][0]} {index['PSNR']} {index['SSIM']}\n")
                avg_meters.update(index)

                if i % 1 == 0 and self.epoch % 2 == 0:
                    output_clean,output_reflection,input,target_r,target_t=model.return_output()

                    
                    img_o = torch.cat((input, output_clean, target_t, output_reflection, target_r), dim=0)
                    # print('input size: ',input.shape)
                    # print('output_clean size: ',output_clean.shape) 
                    # print('target_t size: ',target_t.shape)
                    # print('output_reflection size: ',output_reflection.shape)
                    # print('target_r size: ',target_r.shape)

                    save_image(img_o, os.path.join('./eval_result', f'epoch{self.epoch}+{dataset_name}+step{i}.png'))                                

            # 实时更新进度条右侧显示
                val_pbar.set_postfix(
                    ordered_dict={
                        'PSNR': f"{avg_meters['PSNR']:.4f}",
                        'SSIM': f"{avg_meters['SSIM']:.4f}" 
                    },
                    refresh=False
                )      
            val_pbar.close()              

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
            test_pbar = tqdm(
                test_loader,
                desc="Testing",
                total=len(test_loader),
                ncols=100,  # 建议宽度根据指标数量调整
                dynamic_ncols=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )            
            for i, data in enumerate(test_pbar):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))
                test_pbar.update(1)
            test_pbar.close() 

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
