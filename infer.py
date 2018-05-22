import numpy as np
import os
import torch
from tqdm import tqdm

from data import get_data_loader, load_annotations, save_kaggle_submision
from measures import f1_score, reduce_stats, multilabel_stats
from models import model_type_from_model_file, get_model
from thresholds import calculate_optimal_thresholds_one_by_one
from utils import vector_to_index_list

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
      if len(inputs.size()) == 5:
        # 5d tensor. Then several crops to be evaluated for the same sample
        bs, ncrops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        multicrop = True
      else:
        # Single crop scenario.
        multicrop = False

      with torch.set_grad_enabled(False):
        outputs = model(inputs)
        confidences = torch.sigmoid(outputs)

        if multicrop:
          confidences = confidences.view(bs, ncrops, -1).mean(1)

        ret_confidences += list(confidences.cpu().numpy())
        if isinstance(threshold, np.ndarray):
          threshold = torch.from_numpy(threshold.astype(np.float32)).to(device)
        vec_preds = torch.ge(confidences, threshold).type(confidences.type()).cpu().numpy()
        ret_preds += vector_to_index_list(vec_preds)

        # statistics
        batch_stats = multilabel_stats(labels, confidences, threshold=threshold)
        running_stats = tuple([np.add(a, b) for a, b in zip(running_stats, batch_stats)])

      progress_bar.update(batch_size)
      if samples_limit and samples >= samples_limit:
        break

  per_class_scores = f1_score(*running_stats)
  global_scores = f1_score(*reduce_stats(*running_stats))

  return ret_image_ids, ret_labels, ret_preds, ret_confidences, global_scores, per_class_scores

def infer_runner(img_set_folder, model_file, samples_limit=None, tta=False, batch_size=64):
  set_type = img_set_folder.split("/")[-1]
  model_type = model_type_from_model_file(model_file)
  image_dataset, dataloader = get_data_loader(img_set_folder, model_type, set_type, batch_size=batch_size, tta=tta,
                                              use_test_transforms=True)

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
    vec_preds = np.array(confidences) > model.thresholds # updating prediction with new thresholds.
    preds = vector_to_index_list(vec_preds)
    global_scores = f1_score(*reduce_stats(*multilabel_stats(np.array(labels), np.array(confidences), model.thresholds)))
    np.save("thresholds", model.thresholds)
    np.save(model_file + ".thresholds", model.thresholds)

  #if set_type in ['train', 'validation']:
  #  print("Global results for {}. F1: {:.3}, precision: {:.3}, recall: {:.3}".format(set_type, *global_scores))
  #  np.savetxt("{}_per_class_scores.csv".format(set_type),
  #             np.array([image_dataset.class_frequency()] + list(per_class_scores)).T,
  #             header="original_frequency, f1, precision, recall", delimiter=",")

  if (samples_limit is None and set_type in ['validation', 'test']) or (samples_limit > 25000 and set_type == 'train'):
    # Saving results just for full sets inference
    # They can be used for ensembling
    base_path = os.path.dirname(model_file)
    results_file = os.path.join(base_path,
                                "inference_{}_{}.th".format(set_type, "tta" if tta else "no_tta"))
    torch.save({
      "image_ids": image_ids,
      "thresholds": model.thresholds,
      "labels": labels,
      "confidences": confidences,
      "f1": global_scores[0]
    }, results_file)
    performance_file = os.path.join(base_path,
                                "performance_{}_{}.txt".format(set_type, "tta" if tta else "no_tta"))
    with open(performance_file, "w") as f:
      f.write("{:.4}\n".format(global_scores[0]))

  if set_type == 'test':
    save_kaggle_submision("kaggle_submision.csv", image_ids, preds, image_dataset.classes)


class Ensemble:

  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False):
    self.model_files = model_files
    self.set_type = set_type
    self.set_type_for_thresholds = set_type_for_thresholds
    self.tta = tta
    self.set_data = self.load(set_type)
    self.thresholds_data = self.set_data if set_type == set_type_for_thresholds else self.load(set_type_for_thresholds)

  def load(self, set_type):
    '''
    :return: the tuple of numpy arrays (image_ids, labels, confidences, thresholds) of size N, N X L, M x N x L, M x L
    '''
    tta = self.tta

    as_np = lambda x: (np.array(x['image_ids'], dtype=int), np.array(x['labels'], dtype=np.float32), np.array(x['confidences'], dtype=np.float32),
                       x['thresholds'])

    g_img_ids, g_y, ps, gt = None, None, [], []
    for i, model_file in enumerate(self.model_files):
      base_path = os.path.dirname(model_file)
      data_for_inference = os.path.join(base_path,
                                        "inference_{}_{}.th".format(set_type, "tta" if tta else "no_tta"))

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


class MeanEnsemble(Ensemble):

  def __init__(self, model_files, set_type, set_type_for_thresholds='validation', tta=False):
    super().__init__(model_files, set_type, set_type_for_thresholds, tta)
    self.thresholds = None

  def _combine(self, confidences):
    return confidences.mean(axis=0)

  def _calculate_thresholds(self):
    _, labels, confidences, _ = self.thresholds_data
    ensembled_confidences = self._combine(confidences)
    self.thresholds = calculate_optimal_thresholds_one_by_one(labels, ensembled_confidences, slices=250)

  def infer(self):
    self._calculate_thresholds()
    img_ids, labels, confidences, _ = self.set_data
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



