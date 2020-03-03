import math

import numpy as np
import os
import torch
from torch.nn import Module, Parameter
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset
from tqdm import tqdm

import pytorch_patches
from data import load_annotations, save_kaggle_submision
from measures import f1_score, reduce_stats, multilabel_stats, multilabel_stats_from_pred
from thresholds import calculate_optimal_thresholds_one_by_one
from utils import vector_to_index_list


class LearnersData:

  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False):
    self.model_files = model_files
    self.set_type = set_type
    self.set_type_for_thresholds = set_type_for_thresholds
    self.tta = tta
    self.set_data = self.load(set_type)
    self.thresholds_data = self.set_data if set_type == set_type_for_thresholds else self.load(set_type_for_thresholds)

  def load(self, set_type):
    '''
    :return: the tuple of numpy arrays (image_ids, labels, confidences, thresholds) of size N, N X C, L x N x C, L x C
    '''
    tta = self.tta

    as_np = lambda x: (np.array(x['image_ids'], dtype=int), np.array(x['labels'], dtype=np.float32), np.array(x['confidences'], dtype=np.float32),
                       x['thresholds'])

    g_img_ids, g_y, ps, gt = None, None, [], []
    for i, model_file in enumerate(self.model_files):
      base_path = os.path.dirname(model_file)
      data_for_inference = os.path.join(base_path,
                                        "inference_{}_{}.th".format(set_type, "tta" if tta else "no_tta"))
      if not os.path.exists(data_for_inference) and tta:
        print("No TTA path '{}' found. Trying with no tta file".format(data_for_inference))
        data_for_inference = os.path.join(base_path,
                                          "inference_{}_{}.th".format(set_type, "no_tta"))

      l_img_ids, l_y, l_p, l_t = as_np(torch.load(data_for_inference))
      sorting = np.argsort(l_img_ids)
      if g_img_ids is None:
        g_img_ids = l_img_ids[sorting]
        g_y = l_y[sorting]
      else:
        assert np.allclose(l_img_ids[sorting], g_img_ids), "Image id lists not equal for all datasets."
        assert np.allclose(l_y[sorting], g_y), "Label list not equal for all datasets."

      ps.append(l_p[sorting, :])
      gt.append(l_t)

    np_ps, np_gt = np.array(ps), np.array(gt)
    assert np_ps.shape == (len(self.model_files), g_y.shape[0], g_y.shape[1])
    assert np_gt.shape == (len(self.model_files), g_y.shape[1])

    return g_img_ids, g_y, np_ps, np_gt


class MeanEnsemble(LearnersData):

  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False):
    super().__init__(model_files, set_type, set_type_for_thresholds, tta)
    self.thresholds = None

  def _combine(self, confidences):
    c = confidences.mean(axis=0)
    return c

  def _calculate_thresholds(self):
    _, labels, confidences, model_thresholds = self.thresholds_data
    ensembled_confidences = self._combine(confidences)
    self.thresholds = calculate_optimal_thresholds_one_by_one(labels, ensembled_confidences, slices=100)

  def infer(self):
    self._calculate_thresholds()
    img_ids, labels, confidences, set_thresholds = self.set_data
    fused_confidences = self._combine(confidences)

    vec_preds = fused_confidences > self.thresholds
    preds = vector_to_index_list(vec_preds)

    if self.set_type == 'test':
      global_scores = None
      annotations = load_annotations()
      classes = annotations['train']['classes']
      save_kaggle_submision("ensemble_kaggle_submision.csv", img_ids, preds, classes)
    else:
      global_scores = f1_score(
        *reduce_stats(*multilabel_stats(labels, fused_confidences, self.thresholds)))
      print("Ensemble results for {}. F1: {:.4}, precision: {:.4}, recall: {:.4}".format(self.set_type, *global_scores))

    return img_ids, labels, preds, confidences, global_scores


class QuorumEnsemble(LearnersData):

  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False):
    super().__init__(model_files, set_type, set_type_for_thresholds, tta)

  def infer(self):
    img_ids, labels, confidences, _ = self.set_data
    _, _, _, thresholds = self.thresholds_data
    M = confidences.shape[0]
    assert M % 2 == 1, "Number of models for this modality must be odd"
    # confidences: M x N x L, thresholds: M x L
    vec_preds_per_model = confidences > thresholds[:, np.newaxis, :]
    vec_preds = vec_preds_per_model.sum(axis=0) > M // 2
    preds = vector_to_index_list(vec_preds)

    if self.set_type == 'test':
      global_scores = None
      annotations = load_annotations()
      classes = annotations['train']['classes']
      save_kaggle_submision("ensemble_kaggle_submision.csv", img_ids, preds, classes)
    else:
      global_scores = f1_score(
        *reduce_stats(*multilabel_stats_from_pred(labels, vec_preds)))
      print("Ensemble results for {}. F1: {:.4}, precision: {:.4}, recall: {:.4}".format(self.set_type, *global_scores))

    return img_ids, labels, preds, confidences, global_scores


class PositiveMeanEnsemble(MeanEnsemble):
  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False, positive_threshold=0.2):
    super().__init__(model_files, set_type, set_type_for_thresholds, tta)
    self.positive_threshold = positive_threshold

  def _combine(self, confidences):
    positive_mask = confidences >= self.positive_threshold
    positives_per_sample = (positive_mask).sum(axis=0)
    positives_per_sample[positives_per_sample==0] = 1 # Avoiding div by zero problems
    return ((confidences * positive_mask).sum(axis=0) / positives_per_sample).astype(np.float32)


class BestPerClassEnsemble(MeanEnsemble):
  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False, top=1):
    super().__init__(model_files, set_type, set_type_for_thresholds, tta)
    self.top = top
    #self.train_data = self.load("train")
    self._select_best_models(self.thresholds_data)

  def _select_best_models(self, data):
    thresholds = data[3]
    confidences_LNC = data[2]
    labels = data[1]

    confidences_NLC = confidences_LNC.transpose((1, 0, 2))  # N x L x C
    pred_NLC = confidences_NLC > thresholds  # N x L x C

    pred_LNC = pred_NLC.transpose((1, 0, 2))
    #tp = pred_LNC * labels
    fp = pred_LNC * (1 - labels)
    #tn = (1 - pred_LNC) * (1 - labels)
    fn = (1 - pred_LNC) * (labels)

    fails = fp + fn  # L x N x C
    per_class = fails.sum(axis=1)  # L x C

    best_models_for_class = np.argsort(per_class, axis=0)  # L x C
    self.top_best_models = best_models_for_class[0:self.top, :]  # top x C

  def _combine(self, confidences_LNC):
    confidences_NLC = confidences_LNC.transpose((1, 0, 2))  # N x L x C
    n_classes = confidences_NLC.shape[2]
    return confidences_NLC[:, self.top_best_models, np.arange(n_classes)].mean(axis=1)


class MetaLearnerModel(torch.nn.Module):

  def __init__(self, n_classes, n_learners):
    super().__init__()
    self.layer = DotLinear(n_classes, n_learners)

  def forward(self, input):
    return self.layer(input)

  def print_min_per_class(self):
    print("Min per class: {}".format(self.layer.weight.min(1)[0].data))

  def print_sum_per_class(self):
    print("Sum per class: {}".format(self.layer.weight.sum(1).data))

class MetaLearner:

  def __init__(self, learners_data, batch_size=128):
    '''

    :param learners_data: LearnerData object with data from learners (confidences, labels, etc)
    '''
    self.learners_data = learners_data
    # N, N X C, L x N x C, L x C
    image_ids, labels, confidences, thresholds = learners_data.set_data
    n_learners, n_classes = confidences.shape[0], labels.shape[1]
    # Two tensors. First one of size N x C x L (input) and second one with N x C (labels)
    input_tensor = torch.FloatTensor(np.transpose(confidences, [1, 2, 0])) # N x L x C
    labels_tensor = torch.FloatTensor(labels)
    self.dataset = TensorDataset(input_tensor, labels_tensor)
    self.model = MetaLearnerModel(n_classes, n_learners)
    #self.criterion = pytorch_patches.BCEWithLogitsLoss()
    # self.criterion = torch.nn.BCELoss()
    self.criterion = MetaLearnerLoss(self.model.layer.weight)
    super(MetaLearner, self).__init__()
    self.batch_size = 128

  def train(self, num_epochs):
    #optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)#, weight_decay=1e-5)
    optimizer = torch.optim.Adam(self.model.parameters(), 0.001)
    #scheduler = lr_scheduler.StepLR(optimizer, 100, 0.1)

    for epoch in range(num_epochs):

      running_loss = 0
      samples = 0
      best_f1 = 0
      epoch_confidences = []
      for inputs, labels in torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size):
        samples += labels.size()[0]

        optimizer.zero_grad()

        outputs = self.model(inputs).view(labels.size()).clamp(0, 1)
        #confidences = torch.sigmoid(outputs)
        confidences = outputs
        loss = self.criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_confidences += list(confidences.detach().cpu().data.numpy())
        running_loss += loss.item() * inputs.size(0)

      epoch_loss = running_loss / samples
      print("Loss {:.5}".format(epoch_loss))
      #for name, param in self.model.named_parameters():
      #  print(name)
      #  print(param)

      #scheduler.step(epoch)

      if (epoch + 1) % 10 == 0:
        f1, thresholds = self.eval(epoch_confidences, epoch_loss, epoch)
        #self.model.print_min_per_class()
        #self.model.print_sum_per_class()

        if f1 < best_f1:
          best_f1 = f1
          model_path = "meta_model_best.pth.tar"
          print("Saving model with F1 {} to '{}'".format(best_f1, model_path))
          torch.save({
            'model': self.model.state_dict(),
            'thresholds': thresholds,
            'f1': f1
          }, model_path)

  def eval(self, confidences, epoch_loss, epoch):
    epoch_labels = self.learners_data.set_data[1]
    thresholds = calculate_optimal_thresholds_one_by_one(epoch_labels, confidences, slices=250)
    #thresholds = 0.5
    f1, precision, recall = \
      f1_score(*reduce_stats(
        *multilabel_stats(np.array(epoch_labels, dtype=np.float32), np.array(confidences, dtype=np.float32),
                          threshold=thresholds)))
    print('epoch:{}, loss: {:.4f} F1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch,
      epoch_loss, f1, precision, recall))
    return f1, thresholds


class DotLinear(Module):

  def __init__(self, classes, learners):
    super(DotLinear, self).__init__()
    self.classes = classes
    self.learners = learners
    self.weight = Parameter(torch.Tensor(classes, learners))
    self.bias = Parameter(torch.Tensor(classes, learners))
    self.reset_parameters_2()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    self.bias.data.uniform_(-stdv, stdv)

  def reset_parameters_2(self):
    self.weight.data = torch.ones((self.classes, self.learners)) / self.learners
    self.bias.data = torch.zeros((self.classes, self.learners))

  def forward(self, input):
    #return (input * self.weight + self.bias).sum(2) / self.learners
    return (input * self.weight).sum(2)

  def extra_repr(self):
    return 'classes={}, learners={}'.format(
      self.classes, self.learners
    )


class MetaLearnerLoss(Module):
  ''' BCE loss that enforces that the sum of weights for all learners are 1. '''
  def __init__(self, weights, norm_weight=1.):
    super(MetaLearnerLoss, self).__init__()
    self.bceloss = torch.nn.BCELoss()
    self.weights = weights
    self.norm_weight = norm_weight

  def forward(self, output, labels):
    classes, learners = self.weights.size()
    return self.bceloss(output, labels) + self.norm_weight * torch.abs((torch.ones(classes) - self.weights.sum(1))).mean()
