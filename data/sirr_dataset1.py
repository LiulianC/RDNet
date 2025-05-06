import torch
import os
from dataset.image_folder import make_dataset, default_loader
from PIL import Image
from utils.util import make_power
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Sampler

#"self."的变量是这个类的全局变量    
class SIRRDataset(data.Dataset):
    def __init__(self, opts, train):
        self.opts = opts
        # data root 
        self.data_root = opts.data_root
        self.gen_scenery_data_dir =  os.path.join(self.data_root, opts.gen_scenery_data_dir)
        self.scenery_gen_data = []
        self.gen_tissue_data_dir = os.path.join(self.data_root, opts.gen_tissue_data_dir)
        self.tissue_gen_data = []
        self.real_data_dir = os.path.join(self.data_root, opts.real_data_dir)
        self.real_data=[]

        self.test_data_dir = os.path.join(self.data_root, opts.test_data_dir)
        self.train = train
        self.test_data = []
        self.to_tensor = transforms.ToTensor()
        self.data = [] 
        self.train_data = []
        self.label2 = []
        if train:
            # Load synthetic data
            for file in os.listdir(self.gen_scenery_data_dir):
                if file.endswith('-input.png'):
                    label_file = file.replace('-input.png', '-label1.png')
                    self.scenery_gen_data.append((os.path.join(self.gen_scenery_data_dir, file), os.path.join(self.gen_scenery_data_dir, label_file)))
                    label2_file = file.replace('-input.png', '-label2.png')
                    self.label2.append((os.path.join(self.gen_scenery_data_dir, file), os.path.join(self.gen_scenery_data_dir, label2_file)))
            
            # Load real data
            for file in os.listdir(self.gen_tissue_data_dir):
                if file.endswith('-input.png'):
                    label_file = file.replace('-input.png', '-label1.png')
                    self.tissue_gen_data.append((os.path.join(self.gen_tissue_data_dir, file), os.path.join(self.gen_tissue_data_dir, label_file)))
                    label2_file = file.replace('-input.png', '-label2.png')
                    self.label2.append((os.path.join(self.gen_tissue_data_dir, file), os.path.join(self.gen_tissue_data_dir, label2_file)))

            for file in os.listdir(self.real_data_dir):
                if file.endswith('-input.png'):
                    label_file = file.replace('-input.png', '-label1.png')
                    self.real_data.append((os.path.join(self.real_data_dir, file), os.path.join(self.real_data_dir, label_file)))
                    label2_file = file.replace('-input.png', '-label2.png')
                    self.label2.append((os.path.join(self.real_data_dir, file), os.path.join(self.real_data_dir, label2_file)))

            print('Inference process start. Total train images num: {}'.format(len(self.scenery_gen_data)+len(self.tissue_gen_data)+len(self.real_data)))

            self.train_data = self.scenery_gen_data + self.tissue_gen_data + self.real_data
            self.data = self.train_data
        else:

            for file in os.listdir(self.test_data_dir):
                if file.endswith('-input.png'):
                    label_file = file.replace('-input.png', '-label1.png')
                    self.test_data.append((os.path.join(self.test_data_dir, file), os.path.join(self.test_data_dir, label_file)))
                    label2_file = file.replace('-input.png', '-label2.png')
                    self.label2.append((os.path.join(self.test_data_dir, file), os.path.join(self.test_data_dir, label2_file)))
            
            self.data = self.test_data
            print('Inference process start. Total test images num: {}'.format(len(self.test_data)))
        
                    
    def __len__(self):
        return len(self.data)
    
    def train_size(self): # 获取合成数据的长度
        if self.train:
            return len(self.scenery_gen_data),len(self.tissue_gen_data),len(self.real_data)

    def __getitem__(self, idx):
        input_path, label_path = self.data[idx]
        input_image = Image.open(input_path).convert('RGB')
        label_image = Image.open(label_path).convert('RGB')
        input_image = self.to_tensor(make_power(input_image, base=8))
        label_image = self.to_tensor(make_power(label_image, base=8))

        _, label2_path = self.label2[idx]
        if not os.path.exists(label2_path):
            label2_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        else:
            label2_image = Image.open(label2_path).convert('RGB')
        label2_image = self.to_tensor(make_power(label2_image, base=8))
        return {'I': input_image, 'T': label_image, 'R': label2_image}







# dataloader的抽样方法自定义
class CustomSampler(Sampler):
    def __init__(self, scenery_gen_size, tissue_gen_size, real_size, scenery_gen_samples, tissue_gen_samples, real_samples):
        self.scenery_gen_size = scenery_gen_size
        self.tissue_gen_size = tissue_gen_size
        self.real_size = real_size
        self.scenery_gen_samples = scenery_gen_samples
        self.tissue_gen_samples = tissue_gen_samples
        self.real_samples = real_samples

    def __iter__(self):
        # 生成合成风景的随机索引
        synthetic_indices = torch.randperm(self.scenery_gen_size)[:self.scenery_gen_samples]
        # 生成合成血管的随机索引，并转换为全局索引
        tissue_indices = torch.randperm(self.tissue_gen_size)[:self.tissue_gen_samples] + self.scenery_gen_size
        # 生成真实数据的随机索引，并转换为全局索引
        real_indices = torch.randperm(self.real_size)[:self.real_samples] + self.scenery_gen_size + self.tissue_gen_size
        # 合并并打乱索引
        combined_indices = torch.cat([synthetic_indices, tissue_indices, real_indices])
        combined_indices = combined_indices[torch.randperm(len(combined_indices))]
        return iter(combined_indices.tolist())

    def __len__(self):
        return self.scenery_gen_samples + self.tissue_gen_samples + self.real_samples
