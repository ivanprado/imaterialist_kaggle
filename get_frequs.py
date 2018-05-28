import numpy as np
import os

from data import get_data_loader, load_annotations, Imaterialist, class_frequency
from ensemble import LearnersData
import matplotlib.pyplot as plt

annotations = load_annotations()
sets = ['train', 'validation']
data_dir = 'data'
image_datasets, dataloaders = {}, {}
freqs = {}
for set in sets:
  freq = class_frequency(set, annotations)
  freqs[set] = freq
  np.save("freq-{}".format(set), freq)

model = "runs/" + "May24_07-07-00_cs231n-1se_resnext50_32x4d-bs-64-lr0.0006-mom0.9-wd1e-5-cutout4-minscale0.4-rota15-cas" + "/model_best.pth.tar" # 0.6556, PW1

test_data = LearnersData([model], 'test', 'test', False)
preds_LNC = test_data.set_data[2] > test_data.set_data[3]
preds_NC = preds_LNC.reshape(preds_LNC.shape[1], -1)
freq = preds_NC.sum(axis=0) / preds_NC.shape[0]
freqs['test'] = freq
np.save("freq-test", freq)


sorting_set = 'test'
sorting = np.argsort(freqs[sorting_set])

sorted = {}
for set, freq in freqs.items():
  sorted[set] = freq[sorting]

plt.figure(figsize=(10, 10))
for set, freq in sorted.items():
  plt.plot(np.cumsum(freq), label="{} cum_sum".format(set))
  #plt.plot(freq, marker="x", linewidth=0, label=set)
plt.legend()
plt.show()
