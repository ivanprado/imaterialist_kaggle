import numpy as np
import torch

from measures import multilabel_stats, reduce_stats, f1_score, just_f1


def threshold_boundaries(labels, confidences):
  '''Calculates reasonable upper a lower boundaries for a threshold. Optimal threshold
  should be in between the boundaries. Returns a np array of size (num_classes)

  It seems it doesn't work well, better no use that boundaries as start point'''
  positive_confs, negative_confs = np.copy(confidences), np.copy(confidences)
  positive_confs[labels < 0.5] = np.nan
  negative_confs[labels >= 0.5] = np.nan

  positive_mean = np.nanmean(positive_confs, axis=0)
  positive_mean[np.isnan(positive_mean)] = 1
  positive_std = np.nan_to_num(np.nanstd(positive_confs, axis=0))
  negative_mean = np.nan_to_num(np.nanmean(negative_confs, axis=0))
  negative_std = np.nan_to_num(np.nanstd(negative_confs, axis=0))

  upper_boundary = np.clip(positive_mean + positive_std, 0, 1)
  lower_boundary = np.clip(negative_mean - negative_std, 0, 1)

  print("F1 if brackets mean: {:.3f}".format(just_f1(labels, confidences, (upper_boundary + lower_boundary)/2)))

  return upper_boundary, lower_boundary


def calculate_optimal_thresholds_by_brackets(labels, confidences, convergence_speed=0.01, iterations=100, init_boundaries=False):

  with torch.set_grad_enabled(False):
    labels = np.array(labels)
    labels_t = torch.from_numpy(labels)
    confidences = np.array(confidences)
    confidences_t = torch.from_numpy(confidences)
    n_classes = labels.shape[1]

    if init_boundaries:
      upper, lower = threshold_boundaries(labels, confidences)
    else:
      upper = np.ones(n_classes, confidences.dtype)
      lower = np.zeros(n_classes, confidences.dtype)

    def eval(thresholds):
      return (
        f1_score(*multilabel_stats(labels_t, confidences_t, torch.from_numpy(thresholds))))[0]

    def eval_global(thresholds):
      return (
        f1_score(*reduce_stats(*multilabel_stats(labels_t, confidences_t, torch.from_numpy(thresholds)))))[0]

    last_f1 = 0
    f1_increase = 1
    count = 0
    better = None
    better_f1 = 0
    better_iter = 0
    while count < iterations:
      count += 1
      upper_scores = eval(upper)
      lower_scores = eval(lower)
      upper_better_at = upper_scores > lower_scores
      lower_better_at = ~upper_better_at

      distance = upper - lower
      upper -= lower_better_at * distance * convergence_speed
      lower += upper_better_at * distance * convergence_speed

      mid = (lower + upper) / 2.
      new_f1 = eval_global(mid)
      if new_f1 > better_f1:
        better_f1 = new_f1
        better_iter = count
        better = mid
      f1_increase = new_f1 -last_f1
      last_f1 = new_f1

    print("New F1 (brackets): {:.3f} achieved at iter: {}, convergence speed used: {}".format(better_f1, better_iter, convergence_speed))

  return better


def calculate_optimal_thresholds(labels, confidences, slices=1000, old_thresholds=None):

  with torch.set_grad_enabled(False):
    labels = np.array(labels)
    labels_t = torch.from_numpy(labels)
    confidences = np.array(confidences)
    confidences_t = torch.from_numpy(confidences)
    n_classes = labels.shape[1]

    def eval(thresholds):
      return (
        f1_score(*multilabel_stats(labels_t, confidences_t,
                                   torch.from_numpy(thresholds.astype(np.float32)))))[0]

    def eval_global(thresholds):
      return (
        f1_score(*reduce_stats(*multilabel_stats(labels_t, confidences_t, torch.from_numpy(thresholds.astype(np.float32))))))[0]

    var = np.zeros((n_classes, slices+1), dtype=np.float32)
    for i in range(slices+1):
      th = (np.ones(n_classes) * (i / slices))
      th = th.astype(np.float32)
      var[:, i] = eval(th)

    optimal = (np.argmax(var, axis=1) + (slices - np.argmax(np.flip(var, 1), axis=1))) / (2 * slices)
    if not old_thresholds:
      old_thresholds = np.ones(n_classes, dtype=np.float32)/2

    print("New optimal thresholds, F1: {:.4f}, with old ones F1: {}".format(
      eval_global(optimal), eval_global(old_thresholds/2)))

    return optimal