import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

T, H, W = 5, 240, 424


class Dataset(Dataset):
    def __init__(self, type='train'):
        self.type = type

    def __len__(self):
        return len(os.listdir('Image_inputs/W')) // 2

    def __getitem__(self, index):
        if index >= len(self) - 5:
            index = index - 5

        print(index)

        frames = np.empty((T, H, W, 3), dtype=np.float32)
        holes = np.empty((T, H, W, 1), dtype=np.float32)
        dists = np.empty((T, H, W, 1), dtype=np.float32)

        for i in range(5):
            # rgb
            img_file = os.path.join('Image_inputs', 'W', '{:04d}.jpg'.format(index + i))
            raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
            raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            frames[i] = raw_frame
            # mask
            mask_file = os.path.join('Image_inputs', 'W', '{:04d}.png'.format(index + i))
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))  # cv2.dilate 膨胀操作
            holes[i, :, :, 0] = raw_mask.astype(np.float32)
            # dists
            dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2,
                                                      maskSize=5)  # cv2.distanceTransform（）可以方便地将前景对象提取出来
        # 图片转换成tensor
        frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
        holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
        dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()

        # remove holes 在图片中抠出相对应的洞   [0.4585, 0.456, 0.406]使用了imageNet的平均值
        frames = frames * (1 - holes) + holes * torch.tensor([0.4585, 0.456, 0.406]).view(3, 1, 1, 1)
        # valids area  验证标签，也就是获取图片中被扣掉部分原有的数值分布，用于后续的loss计算
        valids = 1 - holes

        # frames = frames.unsqueeze(0)
        # holes = holes.unsqueeze(0)
        # dists = dists.unsqueeze(0)
        # valids = valids.unsqueeze(0)

        return frames, valids, dists


class Generator(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.images = os.listdir("Image_inputs")

    def generator(self):
        # while True:
        dir = self.images[random.choice(range(len(self.images)))]

        frames = np.empty((T, H, W, 3), dtype=np.float32)
        holes = np.empty((T, H, W, 1), dtype=np.float32)
        dists = np.empty((T, H, W, 1), dtype=np.float32)
        label = np.empty((T, H, W, 3), dtype=np.float32)

        for i in range(5):
            # rgb
            img_file = os.path.join('Image_inputs', dir, 'gt_{:1d}.jpg'.format(i))
            raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
            raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            frames[i] = raw_frame
            label[i] = raw_frame
            # mask
            mask_file = os.path.join('Image_inputs', dir, 'mask_{:1d}.png'.format(i))
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))  # cv2.dilate 膨胀操作
            holes[i, :, :, 0] = raw_mask.astype(np.float32)
            # dists
            dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2,
                                                      maskSize=5)  # cv2.distanceTransform（）可以方便地将前景对象提取出来
        # 图片转换成tensor
        frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
        holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
        dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
        label = torch.from_numpy(np.transpose(label, (3, 0, 1, 2)).copy()).float()

        # remove holes 在图片中抠出相对应的洞   [0.4585, 0.456, 0.406]使用了imageNet的平均值
        frames = frames * (1 - holes) + holes * torch.tensor([0.4585, 0.456, 0.406]).view(3, 1, 1, 1)
        # valids area  验证标签，也就是获取图片中被扣掉部分原有的数值分布，用于后续的loss计算
        valids = 1 - holes

        frames = frames.unsqueeze(0)
        dists = dists.unsqueeze(0)
        valids = valids.unsqueeze(0)
        label = label.unsqueeze(0)

        yield frames, valids, dists, label


class Dataset2(Dataset):
    def __init__(self):
        self.images = os.listdir("image")

    def __len__(self):
        return 1

    def __getitem__(self, index):
        dir = self.images[random.choice(range(len(self.images)))]

        frames = np.empty((T, H, W, 3), dtype=np.float32)
        holes = np.empty((T, H, W, 1), dtype=np.float32)
        dists = np.empty((T, H, W, 1), dtype=np.float32)
        label = np.empty((T, H, W, 3), dtype=np.float32)

        for i in range(5):
            # rgb
            img_file = os.path.join('image', dir, 'gt_{:04d}.jpg'.format(random.choice(range(len(os.listdir(os.path.join('image', dir)))))))
            raw_frame = np.array(Image.open(img_file).convert('RGB')) / 255.
            raw_frame = cv2.resize(raw_frame, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
            frames[i] = raw_frame
            label[i] = raw_frame
            # mask
            mask_file = os.path.join('mask', 'mask_{:04d}.png'.format(random.choice(range(0,52))))
            raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            raw_mask = (raw_mask > 0.5).astype(np.uint8)
            raw_mask = cv2.resize(raw_mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            raw_mask = cv2.dilate(raw_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))  # cv2.dilate 膨胀操作
            holes[i, :, :, 0] = raw_mask.astype(np.float32)
            # dists
            dists[i, :, :, 0] = cv2.distanceTransform(raw_mask, cv2.DIST_L2,
                                                      maskSize=5)  # cv2.distanceTransform（）可以方便地将前景对象提取出来
            # 图片转换成tensor
        frames = torch.from_numpy(np.transpose(frames, (3, 0, 1, 2)).copy()).float()
        holes = torch.from_numpy(np.transpose(holes, (3, 0, 1, 2)).copy()).float()
        dists = torch.from_numpy(np.transpose(dists, (3, 0, 1, 2)).copy()).float()
        label = torch.from_numpy(np.transpose(label, (3, 0, 1, 2)).copy()).float()

        # remove holes 在图片中抠出相对应的洞   [0.4585, 0.456, 0.406]使用了imageNet的平均值
        frames = frames * (1 - holes) + holes * torch.tensor([0.4585, 0.456, 0.406]).view(3, 1, 1, 1)
        # valids area  验证标签，也就是获取图片中被扣掉部分原有的数值分布，用于后续的loss计算
        valids = 1 - holes

        return frames, valids, dists, label
