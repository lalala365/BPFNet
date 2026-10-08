import os
import numpy as np
import random
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.utils import *

class R2CObjData(Dataset):

    def __init__(self, data_root, mode='train', shot=5, image_size=352):
        """
        读取查询图像与GT；从 RefFeat_ICON-R 目录读取 ICON 参考特征（.npz）
        将多张参考图（shot>0 或 -1）逐层(r1..r4)做均值聚合，返回四层参考特征。
        """
        print('this is refdataset')
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_root = data_root
        self.shot = shot

        # data_list: [(img_path, gt_path), ...]
        # class_file_list: {class_name: [path_to_npz, ...]}
        self.data_list, self.class_file_list = collect_r2c_data(
            data_root=self.data_root, mode=self.mode
        )

        if self.mode == 'val' and self.shot not in [-1, 0, 5]:
            # 固定验证集的参考样本子集（与原逻辑一致）
            self.record_class_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """
        返回:
          image: Tensor [3,H,W]
          label: Tensor 或 numpy（与原实现保持：train为Tensor/val-test为np）
          ref_feats: tuple( r1,r2,r3,r4 )  # 每个都是 Tensor[64,h,w] 的空间特征图（shot聚合后）
          name: 文件名（无后缀）
        """
        image_path, label_path = self.data_list[index]
        name = image_path.split('/')[-1][:-4]

        # 读查询图像与标签
        image = rgb_loader(image_path)          # PIL
        label = binary_loader(label_path)

        # 训练增强
        if self.mode == 'train':
            image, label = self.aug_data(image=image, label=label)

        # 变换
        image = self.img_transform(image)
        if self.mode == 'train':
            label = self.gt_transform(label)
        else:
            label = np.asarray(label, np.float32)    # 测试/验证保留原mask形式

        # 读取参考特征（支持: shot>0 或 -1；shot==0 表示Baseline）
        if self.shot > 0 or self.shot == -1:
            # 类别名（和原始实现一致的解析方式）
            class_chosen = image_path.split('/')[-1].split('-')[-2]
            file_class_chosen = self.class_file_list[class_chosen]
            num_aux = len(file_class_chosen)

            if self.mode == 'train':
                ref_idx_list = (random.sample(range(num_aux), self.shot)
                                if self.shot > 0 else list(range(num_aux)))
            else:
                ref_idx_list = list(range(num_aux))  # val/test 使用全部

            # 分层累积（逐层做均值）
            sum_r1 = sum_r2 = sum_r3 = sum_r4 = None
            cnt = 0
            for idx in ref_idx_list:
                npz_path = file_class_chosen[idx]
                # 兼容 .npy（旧）与 .npz（新），优先 .npz
                if npz_path.endswith('.npz'):
                    pkg = np.load(npz_path)
                    r1 = torch.from_numpy(pkg['r1'])  # [64,h,w]
                    r2 = torch.from_numpy(pkg['r2'])
                    r3 = torch.from_numpy(pkg['r3'])
                    r4 = torch.from_numpy(pkg['r4'])
                else:
                    # 兼容旧版：单一np.npy特征（退化为对所有层使用同一个）
                    arr = np.load(npz_path)
                    r1 = r2 = r3 = r4 = torch.from_numpy(arr)

                # 累加
                sum_r1 = r1 if sum_r1 is None else (sum_r1 + r1)
                sum_r2 = r2 if sum_r2 is None else (sum_r2 + r2)
                sum_r3 = r3 if sum_r3 is None else (sum_r3 + r3)
                sum_r4 = r4 if sum_r4 is None else (sum_r4 + r4)
                cnt += 1

            # 逐层均值
            ref_r1 = (sum_r1 / cnt).float().contiguous()
            ref_r2 = (sum_r2 / cnt).float().contiguous()
            ref_r3 = (sum_r3 / cnt).float().contiguous()
            ref_r4 = (sum_r4 / cnt).float().contiguous()
            ref_feats = (ref_r1, ref_r2, ref_r3, ref_r4)
        else:
            # Baseline：不使用参考特征
            ref_feats = (-1, -1, -1, -1)

        return image, label, ref_feats, name

    def record_class_files(self):
        """
        1 <= shot < 5, generating record files（与原逻辑保持一致）
        """
        file_path = './data/dataset_{}shot_val.json'.format(self.shot)
        if os.path.exists(file_path):
            print('load from {}...'.format(file_path))
            with open(file_path, 'r') as f:
                self.class_file_list = json.load(f)
        else:
            print('generating {}...'.format(file_path))
            for cate in self.class_file_list.keys():
                cate_file_pairs = self.class_file_list[cate]
                assert len(cate_file_pairs) > self.shot
                rand_idxs = random.sample(range(len(cate_file_pairs)), self.shot)
                self.class_file_list[cate] = [cate_file_pairs[idx] for idx in rand_idxs]
            with open(file_path, 'w') as f:
                json.dump(self.class_file_list, f, indent=4)

    def aug_data(self, image, label):
        image, label = cv_random_flip(image, label)
        image, label = randomCrop(image, label)
        image, label = randomRotation(image, label)
        image = colorEnhance(image)
        label = randomPeper(label)
        return image, label

