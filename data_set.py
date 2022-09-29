from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
import numpy as np
import os
import torch
import random


class MNISTNoisy(datasets.MNIST):
    def __init__(self,asym=False, seed=0):
        super(MNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
        if asym:
            P = np.eye(10)
            n = nosiy_rate

            P[7, 7], P[7, 1] = 1. - n, n
            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n

            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n

            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n

            y_train_noisy = multiclass_noisify(self.targets, P=P, random_state=seed)
            actual_noise = (y_train_noisy != self.targets).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            self.targets = y_train_noisy

        else:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))

        print("Print noisy label generation statistics:")
        for i in range(10):
            n_noisy = np.sum(np.array(self.targets) == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

        return