from __future__ import print_function, division

import datetime
import time

import numpy as np
import random

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os
import models

import pytorch_patches
from data import Imaterialist, load_annotations, get_data_loader
from models import get_resnet_model
from train import sawtooth, Trainer, LRSensitivity

print("Is CUDA available?: {}".format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(int(time.time()))
random.seed(time.time())
np.random.seed(int(time.time()))
annotations = load_annotations()

#model_file = "runs/"+ "May10_14-35-49_cs231n-1resnet18-clr0.0001-1-mom0.9" + "/model_best.pth.tar" # 0.418
#model_file = "runs/"+ "May10_17-30-52_cs231n-1resnet101-bs-64-clr0.0001-0.5-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.491
#model_file = "runs/"+ "May11_05-47-56_cs231n-1resnet101-bs-64-clr5e-6-0.05-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.502
model_file = "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar" # 0.599
model_file = "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar" # 0.603

#model_file=None
#model_file = "runs/"+ "May15_16-31-15_cs231n-1nasnet-bs-64-clr1e-5-0.1-mom0.9-pos-weight3" + "/model_best.pth.tar" # Nasnet finetuning just fc for a little bit
#model_file = "runs/"+ "May15_17-20-35_cs231n-1nasnet-bs-64-clr1e-4-0.01-rmsprop0.9-1-pos-weight3-wd4e-5-fromcell17" + "/model_best.pth.tar" # Nasnet finetuning just fc for a little bit

model_type = "resnet101"
#model_type = "nasnetlarge"

pretrained=False
if not model_file:
  print("No model file. Using pretrained generic model")
  pretrained=True
model = models.get_model(model_type, len(annotations['train']['classes']), model_file=model_file, pretrained=pretrained)
model = model.to(device)

sets = ['train', 'validation', 'test']
data_dir = 'data'
image_datasets, dataloaders = {}, {}
for set in sets:
  folder = os.path.join(data_dir, set)
  id, dl = get_data_loader(folder, model_type, set, annotations, batch_size=64)
  image_datasets[set] = id
  dataloaders[set] = dl

class_names = image_datasets['train'].classes
num_classes = len(class_names)
#class_freq = image_datasets['train'].class_frequency()


# see https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits_with_logits. Balance between recall/precision
#pos_weight=torch.from_numpy(class_freq/(1-class_freq)).type(torch.float).to(device)
pos_weight=torch.from_numpy(np.ones(num_classes) * 3).type(torch.float).to(device)
#pos_weight=None
# Using BCEWithLogitsLoss because it seems to have better numerical stability than
# using MultiLabelSoftMarginLoss or combining a sigmoid con BCELoss
criterion = pytorch_patches.BCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=0)

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.5, momentum=0.9)
#optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.00001, momentum=0.9)
# https://github.com/tensorflow/models/issues/2648#issuecomment-340663699
# https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py
#optimizer_ft = optim.RMSprop(list(model.last_linear.parameters()) + list(model.cell_17.parameters()), lr=0.1, weight_decay=0.00004, alpha=0.9, eps=1, momentum=0.9)
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=2) # for lr testing
lr_f = lambda x: sawtooth(0.0001, 1, 3, x)
lr_f = lambda x: sawtooth(0.01, 1, 3, x)
exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lambda x: 1)

trainer = Trainer("resnet101-bs-64-lr1e-4-mom0.9-wd4e-4-pos-weight3",
                  model,
                  criterion,
                  optimizer_ft,
                  exp_lr_scheduler,
                  dataloaders['train'],
                  dataloaders['validation'],
                  device,
                  samples_limit=25000,
                  validation_samples_limit=5000,
                  thresholds=model.thresholds
                  )
trainer.train_model(1000)

# lr_sensitivity = LRSensitivity(model,
#                                criterion,
#                                dataloaders['train'],
#                                device)
# lr_sensitivity.run("lr_vs_loss_May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3.png")

# TODO: [x] label smoothing https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/losses/losses_impl.py#L706
# TODO: [ ] grad norm https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_

# TODO: Maybe REGULARIZING NEURAL NETWORKS BY PENALIZING CONFIDENT OUTPUT DISTRIBUTIONS instead of label smoothing?

