import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import resnet34, ResNet34_Weights
import wandb
from argparse import ArgumentParser
from IPython.core.display import display_html
from IPython.display import display
import lightning as L
from datamodules import CIFAR10DataModule
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, GradientAccumulationScheduler, StochasticWeightAveraging
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.tuner.tuning import Tuner
from finetuning_scheduler import FinetuningScheduler
from lightning.pytorch.loggers import WandbLogger
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchmetrics.functional import accuracy
from GraphAttenViTBlocks import *
import warnings

# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
# from PIL import Image

# matplotlib_inline.backend_inline.set_matplotlib_formats("svg",
#                                                         "pdf")  # For export
# matplotlib.rcParams["lines.linewidth"] = 2.0
# sns.reset_orig()
warnings.filterwarnings("ignore")

#* define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # main folder name
PATH_DATASETS = os.path.join(CURRENT_DIR, "data/")
CHECKPOINT_PATH = os.path.join(CURRENT_DIR, "checkpoints/")
SAVE_MODELS_PATH = os.path.join(CURRENT_DIR, "save_models/")
LOGS_PATH = os.path.join(CURRENT_DIR, "logs/")

#* wandb settings
wandb.init(anonymous="allow")
# wandb.login(key='b3518f13f1b3184b76d233e2f2b1f7cbef587a1f')
wandb_logger = WandbLogger(project="CIFAR10",
                           log_model="all",
                           save_dir=CHECKPOINT_PATH)

parser = ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--epochs',
                    default=80,
                    type=int,
                    help='max epochs set in Trainer')
parser.add_argument('--model', default='resnet34', type=str, help='model')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--model_type", type=str, default="ResNet18")
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--use_checkpoint", type=str, default=None)
# parser.add_argument("--name", type=str, default="ResNet18")
args = parser.parse_args()

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed_everything(42)

# # For training, we add some augmentation. Networks are too powerful and would overfit.
# train_transforms = transforms.Compose([
# transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

cifar10_dm = CIFAR10DataModule(data_dir=PATH_DATASETS,
                               batch_size=args.batch_size,
                               train_transform=train_transforms,
                               test_transform=test_transforms)


def create_model(model_type=args.model,use_checkpoint=args.use_checkpoint,embed_feats=512, **kwargs):
    # pre-trained on ImageNet
    if "34" in model_type:
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    else:
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    backbone.maxpool = nn.Identity()
    if "resnet" in model_type:
        backbone.fc = nn.Linear(512, args.num_classes)
        return backbone
    if model_type == "resvit18":
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        if use_checkpoint is not None:
            checkpoint = torch.load(use_checkpoint,map_location=lambda storage, loc: storage)
            backbone.load_state_dict(checkpoint['state_dict'])
            for param in backbone.parameters():
                param.requires_grad = False
        model = nn.Sequential(
            backbone,
            nn.Unflatten(1, (512, 4, 4)),
            Conv2dEmbed(512,embed_feats,patch_size=2,width=4,height=4),
            GViTEncoder(embed_feats,heads=8),
            # shape: (batch_size, nodes=4, feats=512)
            nn.Flatten(1),
            # shape: (batch_size, nodes=4*512=2048)
            nn.Linear(2048,args.num_classes)
        )
        return model
    raise ValueError(f"Unknown model type {model_type}")


class LitResnet(LightningModule):

    def __init__(self, 
                 lr=args.lr, 
                 weight_decay=args.weight_decay, 
                 batch_size=args.batch_size, 
                 model_type=args.model,
                 use_checkpoint=args.use_checkpoint,
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()  # auto by wandb
        self.model = create_model(model_type=model_type,use_checkpoint=use_checkpoint,**kwargs)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        # self.log("train_loss", loss)
        self.log_dict({"train_loss": loss}, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds=preds,
                       target=y,
                       task='multiclass',
                       num_classes=args.num_classes)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, logger=True)

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
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        # allow ['LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'ConstantLR', 'LinearLR']
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode="min",
                                      factor=0.2,
                                      patience=6,
                                      min_lr=5e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
        # steps_per_epoch = 45000 // self.hparams.batch_size
        # scheduler_dict = {
        #     "scheduler":
        #     OneCycleLR(
        #         optimizer,
        #         0.1,
        #         epochs=self.trainer.max_epochs,
        #         steps_per_epoch=steps_per_epoch,
        #     ),
        #     "interval":
        #     "step",
        # }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


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
    ModelCheckpoint(monitor="val_acc", mode="max"),
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
    default_root_dir=CHECKPOINT_PATH,
    max_epochs=args.epochs,
    accelerator="auto",
    # clip gradients' global norm to <=0.5 using gradient_clip_algorithm='norm' by default
    gradient_clip_val=0.5,
    # accumulate gradients every k batches as per the scheduling dict
    # accumulate_grad_batches=8,
    # auto_scale_batch_size="power",
    devices='auto',  # default
    logger=wandb_logger,
    callbacks=callbacks,
)
# tuner = Tuner(trainer)

# # ERROR: self._internal_optimizer_metadata[opt_idx]KeyError: 0
# #* Auto-scale batch size by growing it exponentially (default)
# # tuner.scale_batch_size(model, datamodule=cifar10_dm, mode="power")
# # * Auto-scale batch size with binary search
# # tuner.scale_batch_size(model, mode="binsearch")

# #* finds learning rate automatically
# # sets hparams.lr or hparams.learning_rate to that learning rate
# # Run learning rate finder
# #! smaller num_training for faster build
# lr_finder = tuner.lr_find(model, datamodule=cifar10_dm, num_training=50)
# # Results can be found in
# plt.figure(figsize=(5, 5))
# lr_finder.plot(suggest=True)
# plt.savefig(os.path.join('img', "lr_finder.png"), dpi=300)
# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# if isinstance(new_lr, float):
#     model.hparams.lr = new_lr

trainer.fit(model, datamodule=cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)

# Save the best checkpoint
best_checkpoint = trainer.checkpoint_callback.best_model_path
wandb.save(os.path.join(SAVE_MODELS_PATH, best_checkpoint))
# save best model
best_model = model.load_from_checkpoint(best_checkpoint)
torch.save(best_model.state_dict(), os.path.join(SAVE_MODELS_PATH, 'best_model.pt'))
wandb.save(os.path.join(SAVE_MODELS_PATH, 'best_model.pt'))
# model = LitModel.load_from_checkpoint("path/to/checkpoint.ckpt")

metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
del metrics["step"]
metrics.set_index("epoch", inplace=True)
display(metrics.dropna(axis=1, how="all").head())
sns.relplot(data=metrics, kind="line")
plt.savefig(os.path.join('img', "metrics.png"), dpi=300)

wandb.finish()
print("Done!")
