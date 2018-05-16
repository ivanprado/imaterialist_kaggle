import pretrainedmodels
import torch
from torch import nn
import torchvision

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

def enable_params(module, enable):
  for param in module.parameters():
    param.requires_grad = enable

def get_nasnet_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else False)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  # model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

  if model_file:
    print("Loading model from {}".format(model_file))
    model.load_state_dict(torch.load(model_file))

  # Disabling grads for a test by now
  enable_params(model, False)
  enable_params(model.last_linear, True)
  enable_params(model.cell_17, True)

  return model

def get_model(model_name, *kargs, **kwargs):
  model_conf = models[model_name]
  model = model_conf['model_builder'](model_conf['model_class'], *kargs, **kwargs)

  return model

models = {
  'resnet101': {
    'model_class': torchvision.models.resnet101,
    'model_builder': get_resnet_model,
    'input_size': 224,
    #'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
  },
  'nasnetlarge': {
    'model_class': pretrainedmodels.nasnetalarge,
    'model_builder': get_nasnet_model,
    'input_size': 331,  # resize 354
    #'input_range': [0, 1],
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5]
  }
}
