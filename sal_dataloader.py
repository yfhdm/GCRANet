import torch
from skimage import io, transform, color
import numpy as np
import math
from torch.utils.data import Dataset
import cv2


# ==========================dataset load==========================
class RescaleT(object):

    def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, label,edge = sample['image'], sample['label'],sample["edge"]
        # resize the image to (self.output_size, self.output_size) and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (self.output_size[0], self.output_size[1]), mode='constant')
        lbl = transform.resize(label, (self.output_size[0], self.output_size[1]), mode='constant', order=0,
                               preserve_range=True)
        edge = transform.resize(edge, (self.output_size[0], self.output_size[1]), mode='constant', order=0,
                               preserve_range=True)

        return {'image': img, 'label': lbl, "edge":edge}



class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image, label, edge = sample['image'], sample['label'], sample['edge']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        edge = edge[top: top + new_h, left: left + new_w]

        return {'image': image, 'label': label, "edge":edge}



class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):


        image, label,edge = sample['image'], sample['label'],sample['edge']



        tmpLbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)



        if self.flag == 2:
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image

            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                        np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                        np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                        np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                        np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                        np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                        np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # 标准化
            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  # with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # Normalize
            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                        np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                        np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                        np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))
            # Standard
            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.4669) / 0.2437
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.4669) / 0.2437
                # tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                # tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                # tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:

                tmpImg[:, :, 0] = (image[:, :, 0] - 0.4669) / 0.2437
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.4669) / 0.2437
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.4669) / 0.2437
                # tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                # tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                # tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225


        tmpedge = np.zeros(edge.shape)
        if np.max(edge) < 1e-6:
            edge = edge
        else:
            edge = edge / np.max(edge)
        tmpLbl[:, :, 0] = label[:, :, 0]
        tmpedge[:, :, 0] = edge[:, :, 0]
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))
        tmpedge = tmpedge.transpose((2, 0, 1))
        return {'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl),"edge":torch.from_numpy(tmpedge)}

import random
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = cv2.flip(img,1)
        label = cv2.flip(label,1)
    return img, label

class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, is_edge=False,transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform
        self.is_edge=is_edge

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_name_list[idx])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if 0 == len(self.label_name_list):
            label_3 = np.zeros(image.shape)
            edge = np.zeros(image.shape)
        else:
            label_3 = cv2.imread(self.label_name_list[idx])
            # image, label_3 = cv_random_flip(image, label_3)
            edge = cv2.Canny(label_3, 100, 200)
            kernel = np.ones((5, 5), np.uint8)
            edge = cv2.dilate(edge, kernel, iterations=1)
            edge = edge[:, :, np.newaxis]


        label = np.zeros(label_3.shape[0:2])

        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'image': image, 'label': label, "edge": edge}


        if self.transform:
            sample = self.transform(sample)

        return sample
