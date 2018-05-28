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


#model_file = "runs/"+ "May15_16-31-15_cs231n-1nasnet-bs-64-clr1e-5-0.1-mom0.9-pos-weight3" + "/model_best.pth.tar" # Nasnet finetuning just fc for a little bit
#model_file = "runs/"+ "May15_17-20-35_cs231n-1nasnet-bs-64-clr1e-4-0.01-rmsprop0.9-1-pos-weight3-wd4e-5-fromcell17" + "/model_best.pth.tar" # Nasnet finetuning just fc for a little bit
model_file=None
#model_file = "runs/"+ "May17_16-04-29_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.4507739507786539 Sólo la fc entrenada un poquejo. # INVALIDO
#model_file = "runs/"+ "May17_17-12-16_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-since-block4" + "/model_best.pth.tar" # 0.568 # INVALIDO
model_file = "runs/"+ "May18_07-37-59_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.45
model_file = "runs/"+ "May18_08-53-22_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-from-block3" + "/model_best.pth.tar" # 0.5654
model_file = "runs/"+ "May18_20-13-42_cs231n-1xception-bs-64-lr0.0045-mom0.9-wd1e-5-pos-weight3-from-block3" + "/model_best.pth.tar" # 0.5688
model_file = "runs/"+ "May19_05-43-08_cs231n-1xception-bs-32-lr0.1-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar" # 0.5991
model_file = "runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar" # 0.6036
model_file = "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar" # 0.6038 "Best by now"
#model_file = "runs/"+ "May21_14-13-52_cs231n-1sexception-bs-32-clr0.1-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-just-se-modules" + "/model_best.pth.tar" # 0.6452
#model_file = "runs/"+ "May21_15-14-40_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar" # 0.6496
model_file = "runs/"+ "May21_17-04-17_cs231n-1sexception-bs-38-lr0.1-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6541
#model_file = "runs/"+ "May21_22-09-11_cs231n-1sexception-bs-38-clr0.001-0.01-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6539
#model_file = "runs/"+ "May22_07-17-10_cs231n-1xception-bs-32-lr0.5-mom0.5-wd1e-5-cutout4-minscale0.4-rota15" + "/model_best.pth.tar" # 0.6491, PW1!
#model_file = None
#model_file = "runs/"+ "May23_16-19-52_cs231n-1se_resnext50_32x4d-bs-64-lr0.6-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.6345, PW1!
#model_file = "runs/"+ "May23_21-11-42_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.655, PW1
#model_file = "runs/"+ "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.6556, PW1
#model_file = "runs/"+ "May24_10-35-14_cs231n-1se_resnext50_32x4d-bs-64-clr0.6-0.06-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar" # 0.6483, PW1
#model_file = "runs/"+ "May24_13-04-17_cs231n-1se_resnext50_32x4d-bs-64-lr0.06-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar" # 0.6525, PW1
#model_file = "runs/"+ "May24_16-06-22_cs231n-1se_resnext50_32x4d-bs-64-lr0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-label-smoothing0.1" + "/model_best.pth.tar" # 0.6528, PW1
#model_file = "runs/"+ "May28_10-50-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes50-trainval" + "/model_best.pth.tar" # 0.660, PW1
#model_file = "runs/"+ "May28_16-26-56_cs231n-1se_resnext50_32x4d-bs-64-clr0.06-0.006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar" # 0.6626, PW1
#model_file = "runs/"+ "May28_19-19-15_cs231n-1se_resnext50_32x4d-bs-64-clr0.0006-0.00006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval" + "/model_best.pth.tar" # 0.6633, PW1




#model_type = "resnet101"
#model_type = "nasnetlarge"
#model_type = "xception"
model_type = "sexception"
#model_type = "se_resnext50_32x4d"

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
  id, dl = get_data_loader(folder, model_type, set, annotations, batch_size=38)
  image_datasets[set] = id
  dataloaders[set] = dl

class_names = image_datasets['train'].classes
num_classes = len(class_names)
#class_freq = image_datasets['train'].class_frequency()


# see https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits_with_logits. Balance between recall/precision
#pos_weight=torch.from_numpy(class_freq/(1-class_freq)).type(torch.float).to(device)
#pos_weight=torch.from_numpy(np.ones(num_classes) * 3).type(torch.float).to(device)
pos_weight=None
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
#optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001) # resnet
#optimizer_ft = optim.SGD(model.parameters_to_train, lr=0.5, momentum=0.5, weight_decay=1e-5) # xception
optimizer_ft = optim.SGD(model.parameters_to_train, lr=0.1, momentum=0.9, weight_decay=1e-5) # se_resnext50_32x4d

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=2) # for lr testing
lr_f = lambda x: sawtooth(0.0001, 1, 3, x)
lr_f = lambda x: sawtooth(0.1, 1, 1, x)
lr_f = lambda x: [1, 1, 0.1, 0,1, 0.01][x%5]
exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lr_f)#lambda x: 1)

trainer = Trainer("sexception-bs-38-clr0.1-0.01-0.001-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas-best-classes25-trainval",
                  model,
                  criterion,
                  optimizer_ft,
                  exp_lr_scheduler,
                  dataloaders['train'],
                  dataloaders['validation'],
                  device,
                  samples_limit=25000,
                  validation_samples_limit=5000,
                  #samples_limit=64,
                  #validation_samples_limit=64,
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
# TODO: [X] class aware sampling:
# http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2016-ECCV-RelayBP.pdf
# To address this issue, we apply a sampling strategy, named “class-aware
# sampling”, during training. We aim to fill a mini-batch as uniform as possible
# with respect to classes, and prevent the same example and class from always
# appearing in a permanent order. In practice, we use two types of lists, one is
# class list, and the other is per-class image list, i.e., 401 per-class image lists in
# total. When getting a training mini-batch in an iteration, we first sample a class
# X in the class list, then sample an image in the per-class image list of class X.
# When reaching the end of the per-class image list of class X, a shuffle operation
# is performed to reorder the images of class X. When reaching the end of class list,
# a shuffle operation is performed to reorder the classes. We leverage such a classaware
# sampling strategy to effectively tackle the non-uniform class distribution,
# and the gain of accuracy on the validation set is about 0.6%

# TODO: freeze bn parameters for the few iterations. From SENet paper:
# The parameters of all BN layers were frozen for the last few training
# epochs to ensure consistency between training and testing. (g)

# TODO: Create models with different weighting for each class, so that maybe ensembling is superior
# TODO: Test using softmax instead of sigmoids as in https://arxiv.org/pdf/1805.00932.pdf
# TODO: Test square-root sampling as in https://arxiv.org/pdf/1805.00932.pdf
# TODO: Test just considering one label per image, as in https://arxiv.org/pdf/1511.02251.pdf
