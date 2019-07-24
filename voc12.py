import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
import random

class Voc12(data.Dataset):
    def __init__(self, root, split, ignore_label, mean_bgr, augment, base_size, crop_size, scales, flip, year=2012):
        super(Voc12, self).__init__()
        self.year = year
        self.flip = flip
        self.scales = scales
        self.crop_size = crop_size
        self.base_size = base_size
        self.augment = augment
        self.mean_bgr = np.array(mean_bgr)         #用于给image填充padding
        self.ignore_label = ignore_label #用于给label填充padding，ignore_label是忽略掉不计算loss的label，所以可以用它
        self.split = split
        self.root = os.path.join(root, "VOC%d" %self.year)

        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.label_dir = os.path.join(self.root, "SegmentationClass")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = os.path.join(self.root, "ImageSets/Segmentation/%s.txt" %self.split)
            file_list = tuple(open(file_list, "r"))
            file_list = [i.rstrip() for i in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: %s" %self.split)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.root, self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.root, self.label_dir, image_id + '.png')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int32)   # i don't know why it doesn't work like this, but it can be work like below
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        if self.augment:
            image, label = self._augment(image, label)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1)
        return image_id, image.astype(np.float32), label.astype(np.int64)

    def _augment(self, image, label):
        # 1.resize
        h, w = label.shape
        if self.base_size:
            # let the minn(h, w) as the baseline
            if h > w:
                w, h = self.base_size, int(self.base_size * h / w)
            else:
                w, h = int(self.base_size * w / h), self.base_size
        scale_factor = random.choice(self.scales)
        h, w = int(h * scale_factor), int(w * scale_factor)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2.random crop
        h, w = label.shape
        padd_h = max(self.crop_size - h, 0)
        padd_w = max(self.crop_size - w, 0)

        if padd_h > 0 or padd_w > 0:
            image = cv2.copyMakeBorder(image, 0, padd_h, 0 ,padd_w, borderType=cv2.BORDER_CONSTANT, value=self.mean_bgr)
            label = cv2.copyMakeBorder(label, 0, padd_h, 0, padd_w, borderType=cv2.BORDER_CONSTANT, value=self.ignore_label)
        
        h, w = label.shape

        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h : end_h, start_w : end_w]
        label = label[start_h : end_h, start_w : end_w]

        # random flip
        if self.flip:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()
        return image, label

def voc_loader():
    dataset = Voc12(root="/home/iccd/桌面/TX/PRM-pytorch/demo/datasets/VOCdevkit", split='train', ignore_label=255, mean_bgr=(122.675, 116.669, 104.008), augment=True,
        base_size=None, crop_size=321, scales=[0.5, 0.75, 1.0, 1.25, 1.5], flip=True)
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    return data_loader



if __name__ == "__main__":
    loader = voc_loader()
    print(len(loader))
    