import torch
from torch import nn


def get_resnet_model(model_class, num_classes, model_file=None, pretrained=False):
  model_ft = model_class(pretrained=pretrained)

  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  if model_file:
    print("Loading model from {}".format(model_file))
    model_ft.load_state_dict(torch.load(model_file))

  return model_ft

def get_nasnet_model(model_class, num_classes, model_file=None, pretrained=False):
  model_ft = model_class(num_classes=1000, pretrained='imagenet' if pretrained else False)

  num_ftrs = model_ft.last_linear.in_features
  model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  # model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

  if model_file:
    print("Loading model from {}".format(model_file))
    model_ft.load_state_dict(torch.load(model_file))

  return model_ft
