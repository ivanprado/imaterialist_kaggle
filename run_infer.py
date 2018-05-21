from __future__ import print_function, division

import numpy as np
import pretrainedmodels

import torch
from torchvision import models
import matplotlib.pyplot as plt

from data import get_data_loader
from measures import f1_score, reduce_stats, multilabel_stats
from models import get_resnet_model, get_model
from thresholds import calculate_optimal_thresholds_one_by_one
from train import infer
import resource
plt.ion()


def run(img_set_folder, model_file, model_class, model_type, set_type, samples_limit=None, tta=False, batch_size=64):
  image_dataset, dataloader = get_data_loader(img_set_folder, model_type, set_type, batch_size=batch_size, tta=tta)

  class_names = image_dataset.classes

  print("Is CUDA available?: {}".format(torch.cuda.is_available()))
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = get_model(model_type, len(class_names), model_file=model_file)

  model = model.to(device)

  #model.thresholds = np.load("thresholds.npy")

  image_ids, labels, preds, confidences, global_scores, per_class_scores = \
    infer(model, dataloader, device, samples_limit=samples_limit, threshold=model.thresholds)

  # Uncomment for calculate the thresholds for a particular model
  if True and set_type in ['validation', 'train']:
    print("Calculating thresholds on the fly.")
    model.thresholds = calculate_optimal_thresholds_one_by_one(labels, confidences, slices=250, old_thresholds=model.thresholds)
    global_scores = f1_score(*reduce_stats(*multilabel_stats(np.array(labels), np.array(confidences), model.thresholds)))
    np.save("thresholds", model.thresholds)
    np.save(model_file + ".thresholds", model.thresholds)

  if set_type in ['train', 'validation']:
    print("Global results for {}. F1: {:.3}, precision: {:.3}, recall: {:.3}".format(set_type, *global_scores))
    np.savetxt("{}_per_class_scores.csv".format(set_type),
               np.array([image_dataset.class_frequency()] + list(per_class_scores)).T,
               header="original_frequency, f1, precision, recall", delimiter=",")

  torch.save({
    "thresholds": model.thresholds,
    "labels": labels,
    "confidences": confidences
  }, "inference.th")


  if set_type == 'test':
    image_dataset.save_kaggle_submision("kaggle_submision.csv", image_ids, preds)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT before: {}".format(rlimit))
resource.setrlimit(resource.RLIMIT_NOFILE, (40000, rlimit[1]))
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT after: {}".format(rlimit))

img_set_folder = "data/validation"
model_file= "runs/"+ "May11_05-47-56_cs231n-1resnet101-bs-64-clr5e-6-0.05-mom0.9-imgsize-224" + "/model_best.pth.tar" # 0.502
model_file= "runs/"+ "May11_09-34-42_cs231n-1resnet101-bs-64-clr1e-5-0.1-mom0.9-imgsize-224-pos-weight3" + "/model_best.pth.tar" # 0.599
model_file = "runs/"+ "May16_13-38-21_cs231n-1resnet101-bs-64-lr0.01-mom0.9-wd4e-4-pos-weight3" + "/model_best.pth.tar" # 0.603
model_file = "runs/"+ "May17_17-12-16_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-since-block4" + "/model_best.pth.tar" # 0.568 INVALIDO
model_file = "runs/"+ "May17_16-04-29_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.4507739507786539 Sólo la fc entrenada un poquejo. INVALIDO
model_file = "runs/"+ "May18_07-37-59_cs231n-1xception-bs-64-lr0.045-mom0.9-wd1e-5-pos-weight3-just-fc" + "/model_best.pth.tar" # 0.45
model_file = "runs/"+ "May20_08-35-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3" + "/model_best.pth.tar" # 0.6036
model_file = "runs/"+ "May20_22-27-03_cs231n-1xception-bs-32-clr0.01-0.001-mom0.9-wd1e-5-pos-weight3-cutout4-minscale0.4" + "/model_best.pth.tar" # 0.6038

model_type="xception"
model_class=pretrainedmodels.xception

run(img_set_folder, model_file, model_class, model_type, 'validation', samples_limit=1000, tta=False, batch_size=64)
