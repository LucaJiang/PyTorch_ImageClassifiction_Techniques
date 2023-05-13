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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy
from GraphAttenViTBlocks import *
import warnings
# import albumentations as A

# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
# from PIL import Image

warnings.filterwarnings("ignore")

#* wandb settings
# wandb.login(key='b3518f13f1b3184b76d233e2f2b1f7cbef587a1f')
wandb.init(anonymous="allow")

# matplotlib_inline.backend_inline.set_matplotlib_formats("svg",
#                                                         "pdf")  # For export
# matplotlib.rcParams["lines.linewidth"] = 2.0
# sns.reset_orig()

parser = ArgumentParser()
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='max epochs set in Trainer')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--model_type", type=str, default="ResNet18")
parser.add_argument("--patience", type=int, default=3)
# parser.add_argument("--name", type=str, default="ResNet18")
args = parser.parse_args()

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed_everything(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # main folder name
PATH_DATASETS = os.path.join(CURRENT_DIR, "data/")
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "checkpoints/")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")
# BATCH_SIZE = 256 if torch.cuda.is_available() else 64
wandb_logger = WandbLogger(project="CIFAR10",
                           log_model="all",
                           save_dir=CHECKPOINT_PATH)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

# test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])
# # For training, we add some augmentation. Networks are too powerful and would overfit.
# train_transform = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#         transforms.ToTensor(),
#         transforms.Normalize(DATA_MEANS, DATA_STD),
#     ]
# )

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar10_dm = CIFAR10DataModule(data_dir=PATH_DATASETS,
                               batch_size=args.batch_size,
                               train_transform=train_transforms,
                               test_transform=test_transforms)


def create_model(model_type=args.model_type, embed_feats=256, **kwargs):
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
        backbone.layer4 = nn.Identity()
        model = nn.Sequential(
            backbone,
            nn.Unflatten(1, (256, 8, 8)),
            Conv2dEmbed(256,embed_feats,patch_size=4,width=8,height=8),
            GViTEncoder(embed_feats,256,256,heads=4),
            # shape: (batch_size, nodes=4, feats=256+256*4+256=2560)
            nn.Flatten(1),
            # shape: (batch_size, nodes=4*2560=10240)
            nn.LazyLinear(num_classes)
        )
        return model
    raise ValueError(f"Unknown model type {model_type}")


class LitResnet(LightningModule):

    def __init__(self, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, model_type=args.model_type, **kwargs):
        super().__init__(**kwargs)
        self.hparams.lr = lr
        self.hparams.weight_decay = weight_decay
        self.hparams.model_type = model_type
        self.hparams.batch_size = batch_size
        self.save_hyperparameters()  # auto by wandb
        self.model = create_model(model_type, **kwargs)

        # modules = list(self.model.children())[:-2]
        # self.backbone = nn.Sequential(*modules)

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
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    # def configure_optimizers(self):
    #     # We will support Adam or SGD as optimizers.
    #     if self.hparams.optimizer_name == "Adam":
    #         # AdamW is Adam with a correct implementation of weight decay (see here
    #         # for details: https://arxiv.org/pdf/1711.05101.pdf)
    #         optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
    #     elif self.hparams.optimizer_name == "SGD":
    #         optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
    #     else:
    #         assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

    #     # We will reduce the learning rate by 0.1 after 100 and 150 epochs
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        # only allow ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'ConstantLR', 'LinearLR']
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode="min",
                                      factor=0.2,
                                      patience=20,
                                      min_lr=5e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


model = LitResnet()

early_stopping = EarlyStopping('val_loss',
                               patience=args.patience,
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
    # BackboneFinetuning(10, lambda epoch: 0.1 * (0.5**(epoch // 10))),
    # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/finetuning-scheduler.html?highlight=fine%20tuning%20schedule%20provided#
    # FinetuningScheduler(gen_ft_sched_only=True),
]
trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    # clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
    gradient_clip_val=0.5,
    # accumulate gradients every k batches as per the scheduling dict
    # accumulate_grad_batches=8,
    # auto_scale_batch_size="binsearch",
    # auto_lr_find=True,
    # auto_scale_batch_size="power",
    # devices='auto', # default
    logger=wandb_logger,
    # logger=CSVLogger(save_dir="logs/"),
    callbacks=callbacks,
)
# tuner = Tuner(trainer)

# ERROR: self._internal_optimizer_metadata[opt_idx]KeyError: 0
#* Auto-scale batch size by growing it exponentially (default)
# tuner.scale_batch_size(model, mode="power")
#* Auto-scale batch size with binary search
# tuner.scale_batch_size(model, mode="binsearch")

#* finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
# tuner.lr_find(model)
# Run learning rate finder
# lr_finder = tuner.lr_find(model)

# # Results can be found in
# print(lr_finder.results)

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()

# # update hparams of the model
# model.hparams.lr = new_lr

trainer.fit(model, datamodule=cifar10_dm)
trainer.save_checkpoint(CHECKPOINT_PATH + os.sep + "model.ckpt")
trainer.test(model, datamodule=cifar10_dm)

# trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")
# model = LitModel.load_from_checkpoint("path/to/checkpoint.ckpt")

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sns.relplot(data=metrics, kind="line")

wandb.finish()