import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

trainset = datasets.CIFAR10(
    root='.\data',
    train=True,
    download=True,
    transform=transform)

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i in range(10):
    img, _ = trainset[i]
    row = i // 5
    col = i % 5
    axs[row, col].imshow(img.permute(1, 2, 0))
    axs[row, col].set_title(f"Sample {i}")
    axs[row, col].axis('off')
plt.title('Data Augmentation')
plt.tight_layout()
plt.savefig('data_augmentation.png', dpi=300)
plt.show()