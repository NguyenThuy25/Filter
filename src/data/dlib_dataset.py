from re import A
from typing import List, Optional, Tuple

from glob import glob
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
from torchvision.transforms import transforms
import xml.etree.ElementTree as ET
import os
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt


from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
    RandomRotation,
)

# import albumentations as A

class DlibDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.samples = self.load_data(data_dir, label_file)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        img = Image.open(os.path.join(self.data_dir, item['img_dir'])).convert('RGB') # need to convert all img to RGB
        box_left = int(item['box_left'])
        box_top = int(item['box_top'])
        box_width = int(item['box_width'])
        box_height = int(item['box_height'])
        key_points = item['key_points']
        
        # key_points -= np.array([box_left, box_top])
        
        cropped_img = img.crop((box_left, box_top, box_left + box_width, box_top + box_height))
        # key_points /= np.array([box_width, box_height])
        # key_points -= np.array([0.5, 0.5])
        # key_points *= np.array([box_width, box_height])
        # print(img.size)
        return cropped_img, key_points

    def load_data(self, data_dir: str, label_file: str):
        label_dir = os.path.join(data_dir, label_file)
        print(label_dir)
        tree = ET.parse(label_dir) # read in the file with ElementTree
        root = tree.getroot()
        images = root.find('images')
        # print(images)
        samples = []
        for image in images:
            img_dir = image.attrib['file']
            width = image.attrib['width']
            height = image.attrib['height']

            box = image.find('box')
            box_top = float(box.attrib['top'])
            box_left = float(box.attrib['left'])
            box_width = float(box.attrib['width'])
            box_height = float(box.attrib['height'])

            key_points = np.array([[float(kp.attrib['x']), float(kp.attrib['y'])] for kp in box])
            key_points -= np.array([box_left, box_top])
            # key_points /= np.array([box_width, box_height])

            sample = dict(img_dir=img_dir, width=width, height=height,
                          box_top=box_top, box_left=box_left, box_width=box_width, box_height=box_height, key_points=key_points)
            samples.append(sample)
        return samples
    @staticmethod
    def annotate_image(image, key_point) -> Image:
        # self.samples[index]['box_width']
        # key_point += np.array([0.5, 0.5])
        # key_point *= np.array([self.samples[index]['box_width'], self.samples[index]['box_height']])
        draw = ImageDraw.Draw(image)
        for i in range(0, 68):
          draw.ellipse((key_point[i][0] - 2, key_point[i][1] - 2, key_point[i][0] + 2, key_point[i][1] + 2), fill = 'blue', outline = 'blue')
        # image.show()
        return image
    
class TransformDataset(Dataset):
    def __init__(self, dataset: DlibDataset, transform: Optional[Compose] = None):
        self.dataset = dataset
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                A.Resize(224, 224),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, kps = self.dataset[index]
        image = np.array(image)
        # import IPython ; IPython.embed()
        transformed = self.transform(image=image, keypoints=kps)
        image, kps = transformed['image'], transformed['keypoints']
        _, h, w = image.shape
        kps = kps / np.array([w, h]) - 0.5
        return image, kps.astype(np.float32)
    
    def annotate_batch(image: torch.Tensor, keypoints: np.array):
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        def denormalize(x, mean=IMG_MEAN, std=IMG_MEAN) -> torch.Tensor:
            # 3, H, W, B
            tensor = x.clone().permute(1, 2, 3, 0)
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            # B, 3, H, W
            return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2) # clamp tensor into (0,1)
        
        images = denormalize(image)
        img_batch = []
        for img, kp in zip(images, keypoints):
            img = img.permute(1, 2, 0).cpu().numpy()*255
            h, w, _ = img.shape
            # kp = (kp + 0.5).cpu() * np.array([w, h]) #for gpu
            kp = (kp + 0.5) * np.array([w, h])
            img = DlibDataset.annotate_image(image=Image.fromarray(img.astype(np.uint8)), key_point=kp)
            # img_batch.append(Image.fromarray(img.astype(np.uint8)))
            img_batch.append(transforms.ToTensor()(img))

        # fig = plt.figure(figsize=(8,8))
        # for i in range(len(img_batch)):
        #     ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        #     image = img_batch[i]
        #     for j in range(68):
        #         plt.scatter((keypoints[i][j][0].cpu() + 0.5) *224, (keypoints[i][j][1].cpu()+0.5)*224, s=10, marker='.', c='r')
        #     plt.imshow(image)
        # plt.show()
        # print(type(img_batch))
        return torch.stack(img_batch)
    # def annotate_batch(img_batch, keypoints):
    #     fig = plt.figure(figsize=(8,8))
    #     for i in range(len(img_batch)):
    #         ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    #         image = img_batch[i]
    #         for j in range(68):
    #             plt.scatter((keypoints[i][j][0] + 0.5) *224, (keypoints[i][j][1]+0.5)*224, s=10, marker='.', c='r')
    #         plt.imshow(image)
    #     plt.show()
    #     print(type(img_batch))