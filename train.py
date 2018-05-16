import time
from collections import OrderedDict

import copy
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt

from measures import multilabel_stats, reduce_stats, f1_score
from thresholds import calculate_optimal_thresholds_by_brackets, calculate_optimal_thresholds


class Trainer:

  def __init__(self,
               description,
               model,
               criterion,
               optimizer,
               scheduler,
               train_dataloader,
               val_dataloader,
               device,
               samples_limit=None,
               validation_samples_limit=None,
               thresholds=0.5,
               summary_writer=None):
    self.description = description
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.device = device
    self.samples_limit = samples_limit
    self.validation_samples_limit = validation_samples_limit
    self.thresholds = thresholds

    if not summary_writer:
      self.tensorboard = SummaryWriter(comment=description)
      self.running_dir = self.tensorboard.file_writer.get_logdir()
    else:
      self.tensorboard = summary_writer
      self.running_dir = None


  def train_model(self, num_epochs=25):
    board = self.tensorboard
    model = self.model
    scheduler = self.scheduler
    criterion = self.criterion
    optimizer = self.optimizer
    dataloaders = {
      'validation': self.val_dataloader,
      'train': self.train_dataloader
    }
    since = time.time()

    best_model_wts = copy.deepcopy(self.model.state_dict())
    best_f1 = 0.0
    self.global_step = 0
    thresholds = self.thresholds

    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      phases = ['train']
      if self.val_dataloader:
        phases += ['validation']
      for phase in phases:
        phase_start = time.time()
        if phase == 'train':
          scheduler.step()
          board.add_scalars("epoch/optimizer", {
            'lr': scheduler.get_lr()[0]
          }, self.global_step + 1)

          loss, labels, confidences = self._train_epoch()

          calculate_optimal_thresholds_by_brackets(labels, confidences, init_boundaries=False,
                                                                convergence_speed=0.01, iterations=90)
          thresholds = calculate_optimal_thresholds(labels, confidences, slices=1000,
                                                    old_thresholds=(thresholds if isinstance(thresholds, np.ndarray) else None))
          self.thresholds = thresholds

        else:
          image_ids, labels, preds, confidences, global_scores, per_class_scores = \
            infer(model, dataloaders[phase], self.device, samples_limit=self.validation_samples_limit)

        f1, precision, recall = \
          f1_score(*reduce_stats(
            *multilabel_stats(np.array(labels, dtype=np.float32), np.array(confidences, dtype=np.float32),
                              threshold=thresholds)))

        print('{} loss: {:.4f} F1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
          phase, loss, f1, precision, recall))

        # deep copy the model
        if phase == 'validation' and f1 > best_f1:
          best_f1 = f1
          # best_model_wts = copy.deepcopy(model.state_dict())
          model_path = os.path.join(self.running_dir, "model_best.pth.tar")
          thresholds_path = model_path + ".thresholds"
          print("Saving model with F1 {} to '{}'".format(best_f1, model_path))
          torch.save(best_model_wts, model_path)
          print("Saving thresholds to '{}'".format(thresholds_path))
          np.save(thresholds_path, thresholds)

        if phase == 'train':
          board.add_scalars("epoch/loss", {'train': loss}, self.global_step)
        board.add_scalars("epoch/f1", {phase: f1}, self.global_step)
        board.add_scalars("epoch/precision", {phase: precision}, self.global_step)
        board.add_scalars("epoch/recall", {phase: recall}, self.global_step)

        phase_elapsed = time.time() - phase_start
        print("{} phase took {:.0f}s".format(phase, phase_elapsed))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return  # model

  def _train_epoch(self, train=True):
    board = self.tensorboard
    model = self.model
    criterion = self.criterion
    optimizer = self.optimizer

    running_loss = 0.0
    running_stats = (0., 0., 0.)

    epoch_labels = []
    epoch_confidences = []

    # Iterate over data.
    samples = 0
    dataloader_size = len(self.train_dataloader) * self.train_dataloader.batch_size
    estimated_size = min(self.samples_limit, dataloader_size) if self.samples_limit else dataloader_size
    with tqdm(total=estimated_size) as progress_bar:
      for inputs, labels, img_ids in self.train_dataloader:
        if train:
          model.train()  # Set model to training mode
          self.global_step += 1
        else:
          model.eval()  # Set model to evaluate mode

        samples += labels.size()[0]
        epoch_labels += list(labels.data.numpy())
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(train):
          outputs = model(inputs)
          confidences = torch.sigmoid(outputs)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if train:
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        epoch_confidences += list(confidences.detach().cpu().data.numpy())

        progress_bar.update(inputs.size()[0])

        if self.samples_limit and samples >= self.samples_limit:
          break

    epoch_loss = running_loss / samples

    return epoch_loss, epoch_labels, epoch_confidences


class LRSensitivity(Trainer):

  def __init__(self,
               model,
               criterion,
               train_dataloader,
               device):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 1.1)
    self.epochs = 121
    board = BoardCapturer()

    super().__init__("",
                     model,
                     criterion,
                     optimizer,
                     scheduler,
                     train_dataloader,
                     None,
                     device,
                     samples_limit=train_dataloader.batch_size,
                     validation_samples_limit=None,
                     thresholds=0.5,
                     summary_writer=board)

  def run(self, plot_file):
    super().train_model(self.epochs)

    lrs, losses = self.tensorboard.lr_vs_loss()
    plt.plot(lrs, losses)
    plt.title("Learning rate sensitivity")
    plt.ylabel("loss")
    plt.xlabel("learning rate")
    plt.xscale("log")
    plt.savefig(plot_file, bbox_inches='tight')


class BoardCapturer:
  '''Mimics SummaryWritter, but just captures data into self.data'''
  def __init__(self, *kargs, **kwargs):
    self.data = {}

  def add_scalars(self, prefix, values, step):
    for k, v in values.items():
      key = prefix + "/" + k
      self.add_scalar(key, v, step)

  def add_scalar(self, key, value, step):
    if not key in self.data:
      self.data[key] = OrderedDict()
    self.data[key][step] = value

  def a_vs_b(self, key_a, key_b):
    a_d = self.data[key_a]
    b_d = self.data[key_b]

    a_s, b_s = [], []
    for step, a in a_d.items():
      if step in b_d:
        a_s.append(a)
        b_s.append(b_d[step])

    return a_s, b_s

  def lr_vs_loss(self):
    return self.a_vs_b('epoch/optimizer/lr', 'epoch/loss/train')


def infer(model, dataloader, device, threshold=0.5, samples_limit=None):
  running_stats = (0., 0., 0.)

  ret_labels = []
  ret_image_ids = []
  ret_preds = []
  ret_confidences = []
  # Iterate over data.
  samples = 0

  dataloader_size = len(dataloader) * dataloader.batch_size
  estimated_size = min(samples_limit, dataloader_size) if samples_limit else dataloader_size
  with tqdm(total=estimated_size) as progress_bar:
    for inputs, labels, img_ids in dataloader:
      batch_size = inputs.size()[0]
      samples += batch_size
      model.eval()  # Set model to evaluate mode

      ret_image_ids += list(img_ids.data.numpy())

      ret_labels += list(labels.data.numpy())
      labels = labels.to(device)

      inputs = inputs.to(device)

      with torch.set_grad_enabled(False):
        outputs = model(inputs)
        confidences = torch.sigmoid(outputs)

        ret_confidences += list(confidences.cpu().numpy())
        if isinstance(threshold, np.ndarray):
          threshold = torch.from_numpy(threshold.astype(np.float32)).to(device)
        vec_preds = np.argwhere(torch.ge(confidences, threshold).type(confidences.type()).cpu().numpy())
        # See https://stackoverflow.com/a/43094244/3887420
        # group by [0]
        np_idx_preds = np.split(vec_preds[:, 1], np.cumsum(np.unique(vec_preds[:, 0], return_counts=True)[1])[:-1])
        ret_preds += [list(x) for x in np_idx_preds]

        # statistics
        batch_stats = multilabel_stats(labels, confidences, threshold=threshold)
        running_stats = tuple([np.add(a, b) for a, b in zip(running_stats, batch_stats)])

      progress_bar.update(batch_size)
      if samples_limit and samples >= samples_limit:
        break

  per_class_scores = f1_score(*running_stats)
  global_scores = f1_score(*reduce_stats(*running_stats))

  return ret_image_ids, ret_labels, ret_preds, ret_confidences, global_scores, per_class_scores

def sawtooth(min, max, step_size, epoch):
  '''
  Cyclical learning rate function, in form of sawtooth.
  Step_size is the size of the segments of the function. So a complete
  cycle from min to max to min is done in 2 * step_size epochs
  See https://arxiv.org/pdf/1506.01186.pdf
  '''
  segment = epoch // step_size
  rel_epoch = epoch - segment * step_size
  delta = ((max-min) / step_size) * rel_epoch
  if segment % 2 == 1:
    lr = min + delta
  else:
    lr = max - delta
  return lr
