import numpy as np
import torch


def multilabel_stats(y_true, y_pred, threshold=0.5):
  '''
  Given a toch/numpy y_true and y_pred it returns the tuple
  (true_positives, false_positives, false_negatives)
  being each a numpy array of size # of classes
  '''
  with torch.set_grad_enabled(False):
    if isinstance(y_true, np.ndarray):
      y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
      y_pred = torch.from_numpy(y_pred)
    if isinstance(threshold, np.ndarray):
      threshold = torch.from_numpy(threshold)
      threshold = threshold.type(type(y_pred)).to(y_pred.device)

    y_pred = torch.ge(y_pred, threshold).type(y_pred.type())
    true_positives = (y_true * y_pred).sum(0)
    false_positive = ((1-y_true) * y_pred).sum(0)
    false_negatives = (y_true  * (1 - y_pred)).sum(0)
    return (true_positives.cpu().numpy(), false_positive.cpu().numpy(), false_negatives.cpu().numpy())


def multilabel_stats_np(y_true, confidences, threshold=0.5):
  '''
  Given a numpy y_true and confidences it returns the array
  (true_positives, false_positives, false_negatives)

  Of shape 3 x C if len(thresholds.shape)<2
           3 x T x C if len(thresholds.shape)==2
  '''
  if not isinstance(y_true, np.ndarray):
    y_true = np.array(y_true)
  if not isinstance(confidences, np.ndarray):
    confidences = np.array(confidences)
  if not isinstance(threshold, np.ndarray):
    threshold = np.array(threshold)

  if len(threshold.shape) == 2:
    confidences = confidences[:,np.newaxis,:]
    y_true = y_true[:,np.newaxis,:]
    np.repeat(confidences, threshold.shape[0], axis=1)
    y_pred = confidences > threshold

  true_positives = (y_true * y_pred).sum(0)
  false_positive = ((1-y_true) * y_pred).sum(0)
  false_negatives = (y_true  * (1 - y_pred)).sum(0)
  return np.array([true_positives, false_positive, false_negatives])


def multilabel_stats_from_pred(y_true, y_pred):
  true_positives = (y_true * y_pred).sum(0)
  false_positive = ((1-y_true) * y_pred).sum(0)
  false_negatives = (y_true  * (1 - y_pred)).sum(0)
  return np.array([true_positives, false_positive, false_negatives])


def f1_score_np(stats):
  true_positives, false_positives, false_negatives = stats[0], stats[1], stats[2]
  avoiding_div_by_zero = true_positives == 0
  precision = true_positives / (true_positives + false_positives + avoiding_div_by_zero)
  recall = true_positives / (true_positives + false_negatives + avoiding_div_by_zero)
  f1_score = (2 * precision * recall) / (precision + recall + avoiding_div_by_zero)
  return (f1_score, precision, recall)


def reduce_stats(true_positives, false_positives, false_negatives):
  return (true_positives.sum(), false_positives.sum(), false_negatives.sum())


def f1_score(true_positives, false_positives, false_negatives, epsilon=1e-9):
  '''
  With given arguments computes the precision and recall, and finally
  it computes the f1_score with it.
  :return (f1_score, precision, recall)

  See https://web.archive.org/web/20171203024544/https://www.kaggle.com/wiki/MeanFScore
  '''
  print("TP: {} FP:{}, FN:{}".format(true_positives, false_positives, false_negatives))
  avoiding_div_by_zero = true_positives == 0
  precision = true_positives / (true_positives + false_positives + avoiding_div_by_zero)
  recall = true_positives / (true_positives + false_negatives + avoiding_div_by_zero)
  f1_score = (2 * precision * recall) / (precision + recall + avoiding_div_by_zero)
  return (f1_score, precision, recall)


def just_f1_per_class(labels, confidences, thresholds):
  with torch.set_grad_enabled(False):
    labels_t = torch.from_numpy(labels)
    confidences_t = torch.from_numpy(confidences)
    return (
      f1_score(*multilabel_stats(labels_t, confidences_t,
                                 torch.from_numpy(thresholds.astype(np.float32)))))[0]


def just_f1(labels, confidences, thresholds):
  with torch.set_grad_enabled(False):
    labels_t = torch.from_numpy(labels)
    confidences_t = torch.from_numpy(confidences)
    return (
      f1_score(
        *reduce_stats(*multilabel_stats(labels_t, confidences_t, torch.from_numpy(thresholds.astype(np.float32))))))[0]