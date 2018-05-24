import numbers

import torchvision.transforms.functional as F
from random import random


class MultiScaleMultiplesCrops(object):
  """

  Example:
       >>> transform = Compose([
       >>>    MultiScaleMultiplesCrops(size), # this is a list of PIL Images
       >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
       >>> ])
       >>> #In your test loop you can do the following:
       >>> input, target = batch # input is a 5d tensor, target is 2d
       >>> bs, ncrops, c, h, w = input.size()
       >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
       >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
  """

  def __init__(self, size, resizes=[256, 288, 320, 352]):
    self.resizes = resizes
    self.size = size
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
      self.size = size

  def __call__(self, img):
    resizes = [F.resize(img, size) for size in self.resizes]
    all_crops = []
    for crop in resizes:
      all_crops += F.five_crop(crop, self.size)
    # Also including a crop that includes the whole image, even if aspect ratio is not respected.
    all_crops.append(F.resize(img, self.size))
    for i, crop in enumerate(all_crops):
      if random() > 0.5:
        all_crops[i] = F.hflip(crop)
    return tuple(all_crops)

  def __repr__(self):
    return self.__class__.__name__ + '(size={0},resizes={})'.format(self.size, self.resizes)
