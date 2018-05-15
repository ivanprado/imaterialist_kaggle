from __future__ import print_function, division

import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import matplotlib.pyplot as plt
import os

import pytorch_patches
from data import Imaterialist, load_annotations, get_data_loader
from models import get_resnet_model
from train import sawtooth, train_model

plt.ion()

sets = ['train', 'validation', 'test']
data_dir = 'data'
annotations = load_annotations()
image_datasets, dataloaders = {}, {}
for set in sets:
  folder = os.path.join(data_dir, set)
  id, dl = get_data_loader(folder, set, annotations)
  image_datasets[set] = id
  dataloaders[set] = dl

dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
class_freq = image_datasets['train'].class_frequency()

print("Is CUDA available?: {}".format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file= "runs/"+ "May10_14-35-49_cs231n-1resnet18-clr0.0001-1-mom0.9" + "/model_best.pth.tar" # 0.418
model_file= "runs/"+ "May10_17-30-52_cs231n-1resnet101-bs-64-clr0.0001-0.5-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.491
model_file= "runs/"+ "May11_05-47-56_cs231n-1resnet101-bs-64-clr5e-6-0.05-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.502
model_file= "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar" # 0.599
#model_file=None

pretrained=False
if not model_file:
  print("No model file. Using pretrained generic model")
  pretrained=True
model_ft = get_resnet_model(models.resnet101, num_classes, model_file=model_file, pretrained=pretrained)
model_ft = model_ft.to(device)

# see https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits_with_logits. Balance between recall/precision
#pos_weight=torch.from_numpy(class_freq/(1-class_freq)).type(torch.float).to(device)
pos_weight=torch.from_numpy(np.ones(num_classes) * 3).type(torch.float).to(device)
#pos_weight=None
# Using BCEWithLogitsLoss because it seems to have better numerical stability than
# using MultiLabelSoftMarginLoss or combining a sigmoid con BCELoss
criterion = pytorch_patches.BCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=0.1)

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.5, momentum=0.9)
#optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.00001, momentum=0.9)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=2) # for lr testing
lr_f = lambda x: sawtooth(0.0001, 1, 3, x)
exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lr_f)

tensorboard_writer = SummaryWriter(comment="resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3-labs0.1")
logdir = tensorboard_writer.file_writer.get_logdir()

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, tensorboard_writer, dataloaders, device,
                       num_epochs=2500, samples_limit=30000)

# TODO: [x] label smoothing https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/losses/losses_impl.py#L706
# TODO: [ ] grad norm https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_

# TODO: Maybe REGULARIZING NEURAL NETWORKS BY PENALIZING CONFIDENT OUTPUT DISTRIBUTIONS instead of label smoothing?
