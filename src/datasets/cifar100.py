from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR100
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting

import torch
import torchvision.transforms as transforms
import random
import numpy as np


class CIFAR100_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 5, known_outlier_class: int = 3, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 100))
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
        train_set = MyCIFAR100(root=self.root, train=True, transform=transform, target_transform=target_transform,
                              download=True)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(np.array(train_set.targets), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyCIFAR100(root=self.root, train=False, transform=transform, target_transform=target_transform,
                                  download=True)


        self.ds_mean=(0.5071, 0.4867, 0.4408)
        self.ds_std= (0.2675, 0.2565, 0.2761)


class MyCIFAR100(CIFAR100):
    """
    Torchvision CIFAR100 class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyCIFAR100, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        """Override the original method of the CIFAR100 class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.targets[index], int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
