import time

import copy
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from measures import multilabel_stats, reduce_stats, f1_score
from thresholds import calculate_optimal_thresholds_by_brackets


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
               thresholds=0.5):
    self.description = description
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    self.device = device
    self.samples_limit = samples_limit
    self.thresholds = thresholds

    self.tensorboard = SummaryWriter(comment=description)
    self.running_dir = self.tensorboard.file_writer.get_logdir()

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
    global_step = 0
    thresholds = 0.5

    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch, num_epochs - 1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'validation']:
        phase_start = time.time()
        if phase == 'train':
          scheduler.step()
          board.add_scalars("epoch/optimizer", {
            'lr': scheduler.get_lr()[0]
          }, global_step)

          global_step, epoch_loss, epoch_f1, epoch_precision, epoch_recall = train_or_eval_epoc(
            global_step, model, criterion, optimizer, dataloaders[phase], dataloaders['validation'],
            phase == 'train', board, self.device, samples_limit=self.samples_limit, thresholds=thresholds)

        else:
          image_ids, labels, preds, confidences, global_scores, per_class_scores = \
            infer(model, dataloaders[phase], self.device, samples_limit=self.samples_limit)

          thresholds = calculate_optimal_thresholds_by_brackets(labels, confidences, init_boundaries=False,
                                                                convergence_speed=0.01, iterations=100)

          epoch_f1, epoch_precision, epoch_recall = \
            f1_score(*reduce_stats(
              *multilabel_stats(np.array(labels, dtype=np.float32), np.array(confidences, dtype=np.float32),
                                threshold=thresholds)))

        print('{} loss: {:.4f} F1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
          phase, epoch_loss, epoch_f1, epoch_precision, epoch_recall))

        # deep copy the model
        if phase == 'validation' and epoch_f1 > best_f1:
          best_f1 = epoch_f1
          # best_model_wts = copy.deepcopy(model.state_dict())
          model_path = os.path.join(self.running_dir, "model_best.pth.tar")
          thresholds_path = model_path + ".thresholds"
          print("Saving model with F1 {} to '{}'".format(best_f1, model_path))
          torch.save(best_model_wts, model_path)
          print("Saving thresholds to '{}'".format(thresholds_path))
          np.save(thresholds_path, thresholds)

        board.add_scalars("epoch/loss", {
          'train': epoch_loss
        }, global_step)
        board.add_scalars("epoch/f1", {
          phase: epoch_f1
        }, global_step)
        board.add_scalars("epoch/precision", {
          phase: epoch_precision
        }, global_step)
        board.add_scalars("epoch/recall", {
          phase: epoch_recall
        }, global_step)

        phase_elapsed = time.time() - phase_start
        print("{} phase took {:.0f}s".format(phase, phase_elapsed))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best val f1: {:4f}'.format(best_f1))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return  # model

  def train_or_eval_epoc(global_step, model, criterion, optimizer, train_dataloader, val_dataloader, train,
                         tensorboard_writer, device, report_each=300, samples_limit=None, thresholds=0.5):

    running_loss = 0.0
    running_stats = (0., 0., 0.)

    # Iterate over data.
    samples = 0
    dataloader_size = len(train_dataloader) * train_dataloader.batch_size
    estimated_size = min(samples_limit, dataloader_size) if samples_limit else dataloader_size
    with tqdm(total=estimated_size) as progress_bar:
      for inputs, labels, img_ids in train_dataloader:
        if train:
          model.train()  # Set model to training mode
          global_step += 1
        else:
          model.eval()  # Set model to evaluate mode

        samples += labels.size()[0]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(train):
          outputs = model(inputs)
          preds = torch.sigmoid(outputs)
          loss = criterion(outputs, labels)

          # backward + optimize only if in training phase
          if train:
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        batch_stats = multilabel_stats(labels, preds, threshold=thresholds)
        running_stats = tuple([np.add(a, b) for a, b in zip(running_stats, batch_stats)])
        batch_f1, batch_precision, batch_recall = f1_score(*reduce_stats(*batch_stats))

        progress_bar.update(inputs.size()[0])
        if samples_limit and samples >= samples_limit:
          break

    per_class_stats = f1_score(*running_stats)
    prefix = "train/" if train else "validation/"
    tensorboard_writer.add_histogram(prefix + "f1_per_class", per_class_stats[0], global_step)
    tensorboard_writer.add_histogram(prefix + "precision_per_class", per_class_stats[1], global_step)
    tensorboard_writer.add_histogram(prefix + "recall_per_class", per_class_stats[2], global_step)

    epoch_loss = running_loss / samples
    epoch_f1, epoch_precision, epoch_recall = f1_score(*reduce_stats(*running_stats))

    return global_step, epoch_loss, epoch_f1, epoch_precision, epoch_recall

def train_or_eval_epoc(global_step, model, criterion, optimizer, train_dataloader, val_dataloader, train, tensorboard_writer, device, report_each=300, samples_limit=None, thresholds=0.5):

  running_loss = 0.0
  running_stats = (0., 0., 0.)

  # Iterate over data.
  samples = 0
  dataloader_size = len(train_dataloader) * train_dataloader.batch_size
  estimated_size = min(samples_limit, dataloader_size) if samples_limit else dataloader_size
  with tqdm(total=estimated_size) as progress_bar:
    for inputs, labels, img_ids in train_dataloader:
      if train:
        model.train()  # Set model to training mode
        global_step += 1
      else:
        model.eval()  # Set model to evaluate mode

      samples += labels.size()[0]
      inputs = inputs.to(device)
      labels = labels.to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      # track history if only in train
      with torch.set_grad_enabled(train):
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if train:
          loss.backward()
          optimizer.step()

      # statistics
      running_loss += loss.item() * inputs.size(0)
      batch_stats = multilabel_stats(labels, preds, threshold=thresholds)
      running_stats = tuple([np.add(a, b) for a, b in zip(running_stats, batch_stats)])
      batch_f1, batch_precision, batch_recall = f1_score(*reduce_stats(*batch_stats))

      progress_bar.update(inputs.size()[0])
      if samples_limit and samples >= samples_limit:
        break

  per_class_stats = f1_score(*running_stats)
  prefix = "train/" if train else "validation/"
  tensorboard_writer.add_histogram(prefix + "f1_per_class", per_class_stats[0], global_step)
  tensorboard_writer.add_histogram(prefix + "precision_per_class", per_class_stats[1], global_step)
  tensorboard_writer.add_histogram(prefix + "recall_per_class", per_class_stats[2], global_step)

  epoch_loss = running_loss / samples
  epoch_f1, epoch_precision, epoch_recall = f1_score(*reduce_stats(*running_stats))

  return global_step, epoch_loss, epoch_f1, epoch_precision, epoch_recall


def train_model(model, criterion, optimizer, scheduler, tensorboard_writer, dataloaders, device, num_epochs=25, samples_limit=10000):
  since = time.time()

  best_model_wts = copy.deepcopy(model.state_dict())
  best_f1 = 0.0
  global_step = 0
  thresholds = 0.5

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'validation']:
      phase_start = time.time()
      if phase == 'train':
        scheduler.step()
        tensorboard_writer.add_scalars("epoch/optimizer", {
          'lr': scheduler.get_lr()[0]
        }, global_step)

        global_step, epoch_loss, epoch_f1, epoch_precision, epoch_recall = train_or_eval_epoc(
          global_step, model, criterion, optimizer, dataloaders[phase], dataloaders['validation'],
          phase == 'train', tensorboard_writer, device, samples_limit=samples_limit, thresholds=thresholds)

      else:
        image_ids, labels, preds, confidences, global_scores, per_class_scores = \
          infer(model, dataloaders[phase], device, samples_limit=samples_limit)

        thresholds = calculate_optimal_thresholds_by_brackets(labels, confidences, init_boundaries=False,
                                                              convergence_speed=0.01, iterations=100)

        epoch_f1, epoch_precision, epoch_recall = \
          f1_score(*reduce_stats(*multilabel_stats(np.array(labels, dtype=np.float32), np.array(confidences, dtype=np.float32),
                                                   threshold=thresholds)))

      print('{} loss: {:.4f} F1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
        phase, epoch_loss, epoch_f1, epoch_precision, epoch_recall))

      # deep copy the model
      if phase == 'validation' and epoch_f1 > best_f1:
        best_f1 = epoch_f1
        #best_model_wts = copy.deepcopy(model.state_dict())
        model_path = os.path.join(tensorboard_writer.file_writer.get_logdir(), "model_best.pth.tar")
        thresholds_path = model_path + ".thresholds"
        print("Saving model with F1 {} to '{}'".format(best_f1, model_path))
        torch.save(best_model_wts, model_path)
        print("Saving thresholds to '{}'".format(thresholds_path))
        np.save(thresholds_path, thresholds)

      tensorboard_writer.add_scalars("epoch/loss", {
        'train': epoch_loss
      }, global_step)
      tensorboard_writer.add_scalars("epoch/f1", {
        phase: epoch_f1
      }, global_step)
      tensorboard_writer.add_scalars("epoch/precision", {
        phase: epoch_precision
      }, global_step)
      tensorboard_writer.add_scalars("epoch/recall", {
        phase: epoch_recall
      }, global_step)

      phase_elapsed = time.time() - phase_start
      print("{} phase took {:.0f}s".format(phase, phase_elapsed))

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
  print('Best val f1: {:4f}'.format(best_f1))

  # load best model weights
  #model.load_state_dict(best_model_wts)
  return #model

def infer(model, dataloader, device, threshold=0.5, samples_limit=None):
  ## WORK IN PROGRESS
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