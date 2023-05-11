from __future__ import annotations
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


default_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64, num_workers=6, train_transform = default_transform, test_transform = default_transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        self.all_train_set = datasets.CIFAR10(root=self.data_dir, train=True,
                                    download=True, transform=self.train_transform, num_workers=6)
        self.test_set = datasets.CIFAR10(root=self.data_dir, train=False,
                                         download=True, transform=self.test_transform, num_workers=6)
        self.train_set, self.val_set = random_split(self.all_train_set, [45000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)


# TODO CIFAR100DataModule
# class CIFAR100DataModule(CIFAR10):
#     def setup(self, stage=None):
#         self.all_train_set = datasets.CIFAR100(root=self.data_dir, train=True,
#                                     download=True, transform=self.train_transforms, num_workers=self.num_workers)
#         self.test_set = datasets.CIFAR100(root=self.data_dir, train=False,
#                                             download=True, transform=self.test_transforms, num_workers=self.num_workers)
#         self.train_set, self.val_set = random_split(self.all_train_set, [45000, 5000])
