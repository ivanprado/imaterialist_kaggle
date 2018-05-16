import json
import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, sampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import numpy as np

import models

def make_dataset(dir, class_to_idx, img_to_classes, read_labels):
  images = []
  dir = os.path.expanduser(dir)
  for file in sorted(os.listdir(dir)):
    d = os.path.join(dir, file)
    if os.path.isdir(d):
      continue
    if has_file_allowed_extension(d, ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
      img_id = file.split(".")[0]
      target = np.zeros(len(class_to_idx), dtype=np.float32)
      if read_labels:
        idx_with_class = [class_to_idx[c] for c in img_to_classes[img_id]]
        target[idx_with_class] = 1
      item = (d, target, int(img_id))
      images.append(item)

  return images


class Imaterialist(Dataset):

  def __init__(self, dir, annotations, transform=None, target_transform=None,
               loader=default_loader, read_labels=True,) -> None:
    super(Imaterialist).__init__()
    self.dir = dir
    self.read_labels = read_labels
    self.classes = annotations['classes']
    self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}
    self.idx_to_classes = {i: c for i, c in enumerate(self.classes)}
    try:
      self.img_to_classes = annotations['annotations']
    except:
      self.img_to_classes = {}
      read_labels = False
    self.samples = make_dataset(dir, self.classes_to_idx, self.img_to_classes, read_labels)
    if len(self.samples) == 0:
      raise (RuntimeError("Found 0 files in folder: " + dir + "\n"))

    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

  def __getitem__(self, index):
      """
      Args:
          index (int): Index

      Returns:
          tuple: (sample, target) where target is np array with ones at each class_index (Multiclass)
      """
      path, target, img_id = self.samples[index]
      sample = self.loader(path)
      if self.transform is not None:
          sample = self.transform(sample)
      if self.target_transform is not None:
          target = self.target_transform(target)

      return sample, target, img_id

  def __len__(self):
      return len(self.samples)

  def __repr__(self):
      fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
      fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
      fmt_str += '    Root Location: {}\n'.format(self.dir)
      tmp = '    Transforms (if any): '
      fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
      tmp = '    Target Transforms (if any): '
      fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
      return fmt_str

  def class_frequency(self):
    '''
    :return: a np array of size (classes) with frequency of positive ocurrences (between 0 and 1)
    '''
    if not self.read_labels:
      raise(RuntimeError("It is not possible to compute class frequency for datasets with read_lables=False"))
    counts = np.zeros(len(self.classes), dtype=int)
    for path, target, _ in self.samples:
      counts += target > 0.5
    return counts / float(len(self.samples))

  def save_kaggle_submision(self, file, img_ids, preds):
    with open(file, "w") as f:
      f.write("image_id,label_id\n")
      for id, pred in zip(img_ids, preds):
        cpreds = [self.idx_to_classes[cidx] for cidx in pred]
        f.write("{},{}\n".format(id, " ".join(cpreds)))
    print("Kaggle submision file '{}' written".format(file))


def generate_annotations():
  '''
    Transforms train.json, test.json and validation.json
    generating file annotations.json, with structure:
    {
      'classes': [classes],
      'annotations': {
        imgId: [classes]
      }
    }

    WRONG!!! IDs are not unique between train/validation/test
  '''
  data_path = "data"
  with open('%s/train.json' % (data_path)) as json_data:
    train = json.load(json_data)
  with open('%s/test.json' % (data_path)) as json_data:
    test = json.load(json_data)
  with open('%s/validation.json' % (data_path)) as json_data:
    validation = json.load(json_data)
  annotations = train['annotations'] + validation['annotations']
  all_tags = set()
  img_to_classes = {}
  for annotation in annotations:
    all_tags.update(set(annotation["labelId"]))
    img_to_classes[annotation['imageId']] = annotation["labelId"]
  classes = sorted(list(all_tags))
  class_to_idx = {c: i for i, c in enumerate(classes)}
  idx_to_class = {i: c for i, c in enumerate(classes)}
  transformed_annotations = {
    'classes': classes,
    'annotations': img_to_classes
  }
  with open('%s/annotations.json' % (data_path), "w") as f:
    json.dump(transformed_annotations, f, indent=2)


def load_annotations():
  '''
  :return:
  {'train'/'test'/'validation': {
    'classes': ...,
    'images': ....,
    'annotations': .... # if proceed.
  }
  }
  '''

  data_path = "data"

  with open('%s/classes.json' % (data_path)) as json_data:
    classes = json.load(json_data)
    print("{} Classes".format(len(classes)))

  ret = {}
  for dataset_name in ['train', 'validation', 'test']:
    ds_ret = {'classes': classes}
    with open('%s/%s.json' % (data_path, dataset_name)) as json_data:
      dataset = json.load(json_data)

    img_ids = []
    for image in dataset['images']:
      img_ids += [image['imageId']]
    ds_ret['images'] = img_ids

    img_to_classes = {}
    total_labels = 0
    if 'annotations' in dataset.keys():
      for annotation in dataset['annotations']:
        img_to_classes[annotation['imageId']] = annotation["labelId"]
        total_labels += len(annotation["labelId"])
      ds_ret['annotations'] = img_to_classes
    ret[dataset_name] = ds_ret

    print("Metadata. {}: {} images with {} labels".format(dataset_name, len(img_ids), total_labels))

  return ret


def imshow(inp, title=None):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)  # pause a bit so that plots are updated

class ChunkSampler(sampler.Sampler):
  """Samples elements sequentially from some offset.
  Arguments:
      num_samples: # of desired datapoints
      start: offset where we should start selecting from
  """

  def __init__(self, num_samples, start=0):
    self.num_samples = num_samples
    self.start = start

  def __iter__(self):
    return iter(range(self.start, self.start + self.num_samples))

  def __len__(self):
    return self.num_samples

def get_data_loader(path, model_type, type='validation', annotations=None):
  model_cfg = models.models[model_type]
  img_size = model_cfg['input_size']
  img_stats = model_cfg['mean'], model_cfg['std']
  dt_test = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(*img_stats)
  ])
  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(img_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(*img_stats)
    ]),
    'validation': dt_test,
    'test': dt_test
  }

  if not annotations:
    annotations = load_annotations()
  image_dataset = Imaterialist(path, annotations[type], data_transforms[type], read_labels=type in ['train', 'validation'])
  dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=64,
                                               shuffle=True, num_workers=7)

  return image_dataset, dataloader


#dataset = Imaterialist("data/train", "data/annotations.json")
#print(dataset.class_frequency())
#print(dataset.class_frequency() * len(dataset))
#print((dataset.class_frequency() * len(dataset)).sum())
#print(len(dataset))

#a = load_annotations()
