from __future__ import print_function, division

import numpy as np

import torch
from torchvision import models
import matplotlib.pyplot as plt

from data import get_data_loader
from models import get_resnet_model
from train import infer

plt.ion()


def run(img_set_folder, model_file, model_class, model_type, set_type, samples_limit=None, tta=False):
  image_dataset, dataloader = get_data_loader(img_set_folder, model_type, set_type, batch_size=2, tta=tta)

  class_names = image_dataset.classes

  print("Is CUDA available?: {}".format(torch.cuda.is_available()))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = get_resnet_model(model_class, len(class_names), model_file=model_file)

  model = model.to(device)

  image_ids, labels, preds, confidences, global_scores, per_class_scores = \
    infer(model, dataloader, device, samples_limit=samples_limit, threshold=model.thresholds)

  if set_type in ['train', 'validation']:
    print("Global results for {}. F1: {:.3}, precision: {:.3}, recall: {:.3}".format(set_type, *global_scores))
    np.savetxt("{}_per_class_scores.csv".format(set_type),
               np.array([image_dataset.class_frequency()] + list(per_class_scores)).T,
               header="original_frequency, f1, precision, recall", delimiter=",")

  # Uncomment for calculate the thresholds for a particular model
  #optimal_thresholds = calculate_optimal_thresholds_by_brackets(labels, confidences, init_boundaries=False, convergence_speed=0.01)
  #np.save(model_file + ".thresholds", optimal_thresholds)

  if set_type == 'test':
    image_dataset.save_kaggle_submision("kaggle_submision.csv", image_ids, preds)



img_set_folder = "data/test"
model_file= "runs/"+ "May11_05-47-56_cs231n-1resnet101-bs-64-clr5e-6-0.05-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.502
model_file= "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar" # 0.599

model_type="resnet101"

run(img_set_folder, model_file, models.resnet101, model_type, 'test', samples_limit=None, tta=True)
