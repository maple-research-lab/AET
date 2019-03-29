from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image, PILLOW_VERSION
import os
import errno
import numpy as np
import sys
import csv
import numbers
import h5py

from pdb import set_trace as breakpoint
from torchvision.transforms.functional import _get_inverse_affine_matrix
import math

# Set the paths of the datasets here.
_IMAGENET_DATASET_DIR = '/root/imagenet'
_PLACES205_DATASET_DIR = '/root/Places205'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(long(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        if self.dataset_name=='imagenet':
            assert(self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.split!='train':
                transforms_list = [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
                else:
                    transforms_list = [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)
        elif self.dataset_name=='places205':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            if self.split!='train':
                transforms_list = [
                    transforms.CenterCrop(224),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
                else:
                    transforms_list = [
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
            self.transform = transforms.Compose(transforms_list)
            self.data = Places205(root=_PLACES205_DATASET_DIR, split=self.split,
                transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class DataLoader(object):
    def __init__(self,
                 dataset,  
                 resample=False, fillcolor=0,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        
        self.matrix_transform = transforms.Compose([
            transforms.Normalize((0.,0.,112.,0.,0.,112.,6.4e-5,6.4e-5),(0.75,0.75,118.,0.75,0.75,118.,7.75e-4, 7.75e-4))
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

        self.resample = resample
        self.fillcolor = fillcolor
        
        with h5py.File('../homography.h5', 'r') as hf:
            self.homography = hf['homography'][:] 

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                width, height = img0.size
                center = (img0.size[0] * 0.5 + 0.5, img0.size[1] * 0.5 + 0.5)

                coeffs = self.homography[random.randint(0,499999)]
                
                kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
                img1 = img0.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)
                
                ori_img = self.transform(img0)
                warped_img = self.transform(img1)
                
                coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)
                coeffs = self.matrix_transform(coeffs)
                coeffs = coeffs.view(8)

                return ori_img, warped_img, coeffs
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==3)
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

if __name__ == '__main__':
    dataset = GenericDataset('imagenet','train',random_sized_crop=False)
    dataloader = DataLoader(dataset, batch_size=8, unsupervised=True, shift=28, scale=[0.8,1.2], resample=Image.BILINEAR, fillcolor=(128,128,128))
    
    ii=0
    for b in dataloader():
        data, data_,label = b
        vutils.save_image(data,
                        './original_image_{}.png'.format(str(ii)),
                        normalize=True)
        vutils.save_image(data_,
                        './warped_image_{}.png'.format(str(ii)),
                        normalize=True)
        print(ii)
        ii += 1
        break