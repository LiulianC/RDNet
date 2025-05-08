import os
from os.path import join

import torch.backends.cudnn as cudnn
import faulthandler
import data.dataset_sir as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import data.new_dataset1 as datasets
import torch.multiprocessing as mp
# import wandb
faulthandler.enable()
opt = TrainOptions().parse()
cudnn.benchmark = True
opt.lambda_gan = 0
opt.display_freq = 1
opt.display_id = 1
opt.display_port = 8097
opt.display_freq = 1
opt.num_subnet = 4

opt.gen_scenery_num = 300 
opt.gen_tissue_num  = 500
opt.real_tissue_num = 1000
opt.model_path = './checkpoints/ytmt_ucs_sirs/ytmt_ucs_sirs_latest.pth'# 如果不要在我基础上训练就注释我
opt.print_networks = False



if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True


datadir = './dataset'
datadir_gen_scenery = join(datadir, 'gen_scenery_data')
datadir_gen_tissue = join(datadir, 'gen_tissue_data_64')
datadir_real_tissue = join(datadir, 'real_data')



# train_dataset = datasets.DSRDataset(
#     datadir_syn, read_fns('data/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True)

train_dataset_gen_scenery = datasets.DSRTestDataset(datadir_gen_scenery, enable_transforms=False, if_align=opt.if_align, real=False)
train_dataset_gen_tissue = datasets.DSRTestDataset(datadir_gen_tissue, enable_transforms=False, if_align=opt.if_align, real=True)
train_dataset_real_tissue = datasets.DSRTestDataset(datadir_real_tissue, enable_transforms=False, if_align=opt.if_align, real=False)

# 输入数据集列表
# train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real, train_dataset_nature], [0.2,0.5,0.3])

# 随机抽样
CustomSampler_train = datasets.CustomSampler(size1=len(train_dataset_gen_scenery), size2=len(train_dataset_gen_tissue), size3=len(train_dataset_real_tissue), 
                                             samples_size1=min(opt.gen_scenery_num,len(train_dataset_gen_scenery)), 
                                             samples_size2=min(opt.gen_tissue_num ,len(train_dataset_gen_tissue)) , 
                                             samples_size3=min(opt.real_tissue_num,len(train_dataset_real_tissue)),
                                             )
train_dataset_fusion = train_dataset_gen_scenery + train_dataset_gen_tissue + train_dataset_real_tissue
train_dataloader_fusion = datasets.DataLoader(train_dataset_fusion, batch_size=opt.batchSize, sampler=CustomSampler_train, shuffle=False, num_workers=0, pin_memory=True)





datadir_test_gen_scenery = join(datadir, 'test_gen_scenery_data')
datadir_test_gen_tissue = join(datadir, 'test_gen_tissue_data')
datadir_test_real_tissue = join(datadir, 'test_real_data')


eval_dataset_gen_scenery = datasets.DSRTestDataset(datadir_test_gen_scenery, if_align=opt.if_align, enable_transforms=False, real=False)
eval_dataset_gen_tissue = datasets.DSRTestDataset(datadir_test_gen_tissue, if_align=opt.if_align, enable_transforms=False, real=False)
eval_dataset_real_tussue = datasets.DSRTestDataset(datadir_test_real_tissue, if_align=opt.if_align, enable_transforms=False, real=False)

eval_dataloader_gen_scenery = datasets.DataLoader(eval_dataset_gen_scenery, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_gen_tissue = datasets.DataLoader(eval_dataset_gen_tissue, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real_tissue = datasets.DataLoader(eval_dataset_real_tussue, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)







"""Main Loop"""
# engine = Engine(opt,eval_dataloader_real,eval_dataloader_solidobject,eval_dataloader_postcard,eval_dataloader_wild)
engine = Engine(opt,eval_dataloader_gen_scenery, eval_dataloader_gen_tissue, eval_dataloader_real_tissue)

result_dir = os.path.join(f'./experiment_blood/{opt.name}/results',mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


# if opt.resume or opt.debug_eval:
#     save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
#     os.makedirs(save_dir, exist_ok=True)
#     engine.eval(eval_dataloader_real, dataset_name='testdata_real20', savedir=save_dir, suffix='real20')

#     engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject', savedir=save_dir,suffix='solidobject')

#     engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard', savedir=save_dir, suffix='postcard')

#     engine.eval(eval_dataloader_wild, dataset_name='testdata_wild', savedir=save_dir, suffix='wild')

# define training strategy
if __name__ == '__main__':
    mp.freeze_support()
    engine.model.opt.lambda_gan = 0
    set_learning_rate(1e-5)
    while engine.epoch <= 40:
        engine.train(train_dataloader_fusion)
