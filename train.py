import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch
# import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
# from torchvision.datasets import CIFAR10
import wandb
from argparse import ArgumentParser
from IPython.core.display import display_html
from IPython.display import display
import lightning as L
from datamodules import CIFAR10DataModule
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, BackboneFinetuning, GradientAccumulationScheduler, StochasticWeightAveraging
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.tuner.tuning import Tuner
from finetuning_scheduler import FinetuningScheduler
from lightning.pytorch.loggers import WandbLogger
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from GraphAttenViTBlocks import *
# import albumentations as A

# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
# from PIL import Image

parser = ArgumentParser()
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='max epochs set in Trainer')
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--model_type", type=str, default="ResNet18")
parser.add_argument("--key", type=str, default="b3518f13f1b3184b76d233e2f2b1f7cbef587a1f")
# parser.add_argument("--name", type=str, default="ResNet18")

args = parser.parse_args()

wandb.login(key=args.key) 
matplotlib_inline.backend_inline.set_matplotlib_formats("svg",
                                                        "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # main folder name
PATH_DATASETS = os.path.join(CURRENT_DIR, "data/")
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "checkpoints/")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
wandb_logger = WandbLogger(project="CIFAR10", log_model="all", save_dir=CHECKPOINT_PATH)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=args.batch_size,
    num_workers=int(os.cpu_count() / 2),
    train_transform=train_transforms,
    test_transform=test_transforms
)


def create_model(model_type=args.model_type, embed_feats=64, **kwargs):
    # pre-trained on ImageNet
    num_classes = 10
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # model = torchvision.models.resnet18(pretrained=True, num_classes=10)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    backbone.maxpool = nn.Identity()
    if model_type == "ResNet18":
        backbone.fc = nn.Linear(512, num_classes)
        return backbone
    if model_type == "ResViT18":
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        backbone.layer3 = nn.Identity()
        backbone.layer4 = nn.Identity()
        model = nn.Sequential(
            backbone,
            nn.Unflatten(1, (128, 16, 16)),
            Conv2dEmbed(128,embed_feats,patch_size=4,width=16,height=16),
            GViTEncoder(embed_feats,64,64,heads=8),
            # shape: (batch_size, nodes=16, feats=64+64*8+64=640)
            nn.Flatten(1),
            # shape: (batch_size, nodes=16*640=10240)
            nn.Linear(10240, num_classes)
        )
        return model
    raise ValueError(f"Unknown model type {model_type}")


class LitResnet(LightningModule):

    def __init__(self, lr=args.lr, weight_decay=args.weight_decay, model_type=args.model_type, **kwargs):
        super().__init__(**kwargs)
        self.hparams.lr = lr
        self.hparams.weight_decay = weight_decay
        self.hparams.model_type = model_type
        self.save_hyperparameters()  # auto by wandb
        self.model = create_model(model_type, **kwargs)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = 5000 // self.hparams.batch_size
        scheduler_dict = {
            "scheduler":
            OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval":
            "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


model = LitResnet()

early_stopping = EarlyStopping('val_loss',
                               patience=3,
                               verbose=True,
                               mode='min')

# till 5th epoch, it will accumulate every 8 batches. From 5th epoch
# till 9th epoch it will accumulate every 4 batches and after that no accumulation
# will happen. Note that you need to use zero-indexed epoch keys here
# accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
callbacks = [
    LearningRateMonitor(logging_interval="step"),
    StochasticWeightAveraging(swa_lrs=1e-2),
    GradientAccumulationScheduler(scheduling={
        0: 8,
        4: 4,
        8: 1
    }),
    TQDMProgressBar(refresh_rate=10),
    early_stopping,
    BackboneFinetuning(10, lambda epoch: 0.1 * (0.5**(epoch // 10))),
    FinetuningScheduler(),
]
trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    gradient_clip_val=
    0.5,  # clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
    devices=1
    if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=wandb_logger,
    # api_key: b3518f13f1b3184b76d233e2f2b1f7cbef587a1f
    # logger=CSVLogger(save_dir="logs/"),
    callbacks=callbacks,
)
tuner = Tuner(trainer)
#* Auto-scale batch size by growing it exponentially (default)
tuner.scale_batch_size(model, mode="power")
#* Auto-scale batch size with binary search
# tuner.scale_batch_size(model, mode="binsearch")

#* finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
tuner.lr_find(model)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

# trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")
# model = LitModel.load_from_checkpoint("path/to/checkpoint.ckpt")

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sns.relplot(data=metrics, kind="line")

wandb.finish()