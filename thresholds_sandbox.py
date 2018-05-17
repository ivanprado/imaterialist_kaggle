import resource

import numpy as np

from data import Imaterialist, load_annotations
from measures import just_f1_per_class
from thresholds import *
from utils import Timer

#freqs = Imaterialist("data/train", load_annotations()['train']).class_frequency()
#np.save("freqs", freqs)
freqs = np.load("freqs.npy")

def avg_f1(labels, confidences, thresholds):
  return just_f1_per_class(labels, confidences, thresholds).mean()

def weighted_avg_f1(labels, confidences, thresholds, weights):
  return np.average(just_f1_per_class(labels, confidences, thresholds), weights=weights)

def global_f1(labels, confidences, thresholds):
  return just_f1(labels, confidences, thresholds)

confidences = np.load("confidences.npy")
labels = np.load("labels.npy")
unique_labels, unique_counts = np.unique(np.where(labels)[1], return_counts=True)
perct = [25, 50, 75]
percentiles = np.percentile(unique_counts, perct)
counts = labels.sum(axis=0)
print("samples {} classes {} appearing classes {}".format(*confidences.shape + (unique_labels.shape[0],)))
print("percentiles {}: {}".format(perct, percentiles))
n_samples, n_classes = confidences.shape
def_thr = np.ones(n_classes) * 0.5

with Timer("Seq") as t:
  seq_thr = calculate_optimal_thresholds(labels, confidences, slices=10)

with Timer("Brakets") as t:
  brack_thr = calculate_optimal_thresholds_by_brackets(labels, confidences, iterations=10)

with Timer("One-by-one") as t:
  obo_thr = calculate_optimal_thresholds_one_by_one(labels, confidences, slices=1000)

print()
for type, thr in [("0.5", def_thr), ("Seq", seq_thr), ("Brackects", brack_thr), ("One-by-one", obo_thr)]:
  print("{} per_class: {}, weighted: {}, global: {}".format(
    type,
    avg_f1(labels, confidences, thr),
    weighted_avg_f1(labels, confidences, thr, freqs),
    global_f1(labels, confidences, thr)))

stacked = np.array([counts/n_samples, seq_thr, brack_thr]).T
pass

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT before: {}".format(rlimit))
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print("LIMIT after: {}".format(rlimit))
