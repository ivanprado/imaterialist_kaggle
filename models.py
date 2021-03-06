import pretrainedmodels
import torch
from pretrainedmodels.models.xception import Xception, pretrained_settings
from torch import nn
import torchvision
import numpy as np
from torch.utils import model_zoo

import sexception


def load_model(model, model_file, strict=True):
  print("Loading model from {}".format(model_file))
  state_dict = torch.load(model_file)
  model.load_state_dict(state_dict, strict=strict)

  try:
    thresholds_file = model_file + ".thresholds.npy"
    thresholds = np.load(thresholds_file)
    print("Thresholds loaded from {}".format(thresholds_file))
  except FileNotFoundError:
    print("Not possible to load thresholds file from {}. Using default 0.5 threshold".format(thresholds_file))
    thresholds = 0.5

  model.thresholds = thresholds


def get_resnet_model(model_class, num_classes, model_file=None, pretrained=False):
  model_ft = model_class(pretrained=pretrained)

  num_ftrs = model_ft.fc.in_features
  model_ft.fc = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  if model_file:
    load_model(model_ft, model_file)

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
    load_model(model, model_file)

  # Disabling grads for a test by now
  enable_params(model, False)
  enable_params(model.last_linear, True)
  enable_params(model.cell_17, True)

  return model

def get_xception_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else False)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  # model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

  all_layers = [
    'conv1',
    'bn1',
    'conv2',
    'bn2',
    'block1',
    'block2',
    'block3',
    'block4',
    'block5',
    'block6',
    'block7',
    'block8',
    'block9',
    'block10',
    'block11',
    'block12',
    'conv3',
    'bn3',
    'conv4',
    'bn4',
    'last_linear'
  ]
  layers_to_train = [
    'block3',
    'block4',
    'block5',
    'block6',
    'block7',
    'block8',
    'block9',
    'block10',
    'block11',
    'block12',
    'conv3',
    'bn3',
    'conv4',
    'bn4',
    'last_linear'
  ]
  #layers_to_train = ['last_linear'] # Just the FC, typically the first stage after pretrained model.
  parameters_to_train = []
  for name, parameter in model.named_parameters():
    if name.split(".")[0] in layers_to_train:
      parameters_to_train.append(parameter)
      parameter.requires_grad = True
    else:
      parameter.requires_grad = False
  model.parameters_to_train = parameters_to_train

  if model_file:
    load_model(model, model_file)
  else:
    model.thresholds = 0.5

  # Disabling grads for a test by now
  #enable_params(model, False)
  #enable_params(model.last_linear, True)

  return model

def get_xception_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else False)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  # model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

  all_layers = [
    'conv1',
    'bn1',
    'conv2',
    'bn2',
    'block1',
    'block2',
    'block3',
    'block4',
    'block5',
    'block6',
    'block7',
    'block8',
    'block9',
    'block10',
    'block11',
    'block12',
    'conv3',
    'bn3',
    'conv4',
    'bn4',
    'last_linear'
  ]
  layers_to_train = [
    'block3',
    'block4',
    'block5',
    'block6',
    'block7',
    'block8',
    'block9',
    'block10',
    'block11',
    'block12',
    'conv3',
    'bn3',
    'conv4',
    'bn4',
    'last_linear'
  ]
  #layers_to_train = ['last_linear'] # Just the FC, typically the first stage after pretrained model.
  #parameters_to_train = []
  #for name, parameter in model.named_parameters():
  #  if name.split(".")[0] in layers_to_train:
  #    parameters_to_train.append(parameter)
  #    parameter.requires_grad = True
  #  else:
  #    parameter.requires_grad = False
  #model.parameters_to_train = parameters_to_train
  # Training everything
  model.parameters_to_train = model.parameters()

  if model_file:
    load_model(model, model_file)
  else:
    model.thresholds = 0.5

  # Disabling grads for a test by now
  #enable_params(model, False)
  #enable_params(model.last_linear, True)

  return model

def get_sexception_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else False)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  # Replacing fixed global pooling with adaptive so that variable input image size is allowed
  # model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

  # Just train the SE modules... to initialize to correct values.
  # parameters_to_train = []
  # for name, parameter in model.named_parameters():
  #   if "se_module" in name:
  #     parameters_to_train.append(parameter)
  #     parameter.requires_grad = True
  #   else:
  #     parameter.requires_grad = False
  # model.parameters_to_train = parameters_to_train
  # Training everything
  model.parameters_to_train = model.parameters()

  if model_file:
    load_model(model, model_file, strict=False)
  else:
    model.thresholds = 0.5

  # Disabling grads for a test by now
  #enable_params(model, False)
  #enable_params(model.last_linear, True)

  return model


def get_se_resnext50_32x4d_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else None)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  model.parameters_to_train = model.parameters()

  if model_file:
    load_model(model, model_file, strict=False)
  else:
    model.thresholds = 0.5

  return model


def get_se_resnext101_32x4d_model(model_class, num_classes, model_file=None, pretrained=False):
  model = model_class(num_classes=1000, pretrained='imagenet' if pretrained else None)

  num_ftrs = model.last_linear.in_features
  model.last_linear = nn.Linear(num_ftrs, num_classes)
  model.parameters_to_train = model.parameters()

  if model_file:
    load_model(model, model_file, strict=False)
  else:
    model.thresholds = 0.5

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
  }, 'xception': {
    'model_class': pretrainedmodels.xception,
    'model_builder': get_xception_model,
    'input_size': [3, 299, 299],
    'input_range': [0, 1],
    'input_size': 299,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    #'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
  }, 'sexception': {
    'model_class': sexception.sexception,
    'model_builder': get_sexception_model,
    'input_size': [3, 299, 299],
    'input_range': [0, 1],
    'input_size': 299,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    #'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
  },'se_resnext50_32x4d': {
    'model_class': pretrainedmodels.se_resnext50_32x4d,
    'model_builder': get_se_resnext50_32x4d_model,
    'input_size': 224,
    'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    }, 'se_resnext101_32x4d': {
    'model_class': pretrainedmodels.se_resnext101_32x4d,
    'model_builder': get_se_resnext101_32x4d_model,
    'input_size': 224,
    'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    }
}


def model_type_from_model_file(model_file):
  for model_type in models.keys():
    if "1" + model_type in model_file:
      return model_type
  return None