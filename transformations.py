import numbers

import torchvision.transforms.functional as F


class MultiScaleFiveCrop(object):
  """
  Resizes the images to 256, 288, 320 and 352.
  Then take the five Crop as in FiveCrop transformation for each of them.
  In total, 4 * 5 = 20 crops per image.
  Crop the given PIL Image into four corners and the central crop

  Example:
       >>> transform = Compose([
       >>>    MultiScaleFiveCrop(size), # this is a list of PIL Images
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
    resizes = [F.resize(size) for size in self.resizes]
    all_crops = []
    for crop in resizes:
      all_crops.append(F.five_crop(crop, self.size))
    return all_crops

  def __repr__(self):
    return self.__class__.__name__ + '(size={0},resizes={})'.format(self.size, self.resizes)
