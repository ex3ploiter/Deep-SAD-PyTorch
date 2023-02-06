from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random
import numpy as np

import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import is_image_file

from typing import Optional, Callable, List, Tuple, Dict, Any

from torch.utils.data import Dataset
import glob



class MVTec_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 15))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # CIFAR-10 preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyMVTec(root=self.root, train=True, transform=transform, target_transform=target_transform)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.labels), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyMVTec(root=self.root, train=False, transform=transform, target_transform=target_transform)




class MyMVTec(Dataset):
    def __init__(self, root, normal_class, transform=None, target_transform=None, train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.train=train
        # root=os.path.join(root,'mvtec_anomaly_detection')
        
        mvtec_labels=['bottle' , 'cable' , 'capsule' , 'carpet' ,'grid' , 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood','zipper']
        category=mvtec_labels[normal_class]

        if train:
            self.image_files = glob.glob(
                os.path.join(root, '*', "train", "good", "*.png"))
        else:
          image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          
          self.image_files = normal_image_files+anomaly_image_files

        
        self.labels=[]
        for pth in self.image_files:
            img_class=pth.split('/')[3]
            img_class_idx=mvtec_labels.index(img_class)
            self.labels.append(img_class_idx)
        

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        semi_target=int(self.semi_targets[index])
        
        target=self.labels[index]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.train == False:
            if os.path.dirname(image_file).endswith("good"):
                target = 0
            else:
                target = 1
            
        
        
        return image, target, index
        

    def __len__(self):
        return len(self.image_files)
    
    