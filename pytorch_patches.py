import numpy as np
import random

from torch.nn.modules.loss import _Loss


# From https://github.com/pytorch/pytorch/pull/6856
from torch.utils.data.sampler import Sampler


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True, pos_weight=None):
  r"""Function that measures Binary Cross Entropy between target and output
  logits.
  See :class:`~torch.nn.BCEWithLogitsLoss` for details.
  Args:
      input: Tensor of arbitrary shape
      target: Tensor of the same shape as input
      weight (Tensor, optional): a manual rescaling weight
              if provided it's repeated to match input tensor shape
      pos_weight (Tensor, optional): a weight of positive examples.
              Must be a vector with length equal to the number of classes.
      size_average (bool, optional): By default, the losses are averaged
              over observations for each minibatch. However, if the field
              :attr:`size_average` is set to ``False``, the losses are instead summed
              for each minibatch. Default: ``True``
      reduce (bool, optional): By default, the losses are averaged or summed over
              observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
              is ``False``, returns a loss per input/target element instead and ignores
              :attr:`size_average`. Default: ``True``
  Examples::
       >>> input = torch.randn(3, requires_grad=True)
       >>> target = torch.FloatTensor(3).random_(2)
       >>> loss = F.binary_cross_entropy_with_logits(input, target)
       >>> loss.backward()
  """
  if not (target.size() == input.size()):
    raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

  max_val = (-input).clamp(min=0)

  if pos_weight is None:
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
  else:
    log_weight = 1 + (pos_weight - 1) * target
    loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

  if weight is not None:
    loss = loss * weight

  if not reduce:
    return loss
  elif size_average:
    return loss.mean()
  else:
    return loss.sum()

class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    The loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],
    where :math:`N` is the batch size. If reduce is ``True``, then
    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.
    It's possible to trade off recall and precision by adding weights to positive examples.
    In this case the loss can be described as:
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ p_n t_n \cdot \log \sigma(x_n)
        + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right],
    where :math:`p_n` is the positive weight of class :math:`n`.
    :math:`p_n > 1` increases the recall, :math:`p_n < 1` increases the precision.
    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then `pos_weight` for the class should be equal to math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains math:`3\times 100=300` positive examples.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is False, returns a loss per input/target element instead and ignores
            size_average. Default: True
     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input
     Examples::
        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.FloatTensor(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, reduce=True, pos_weight=None, label_smoothing=0):
      super(BCEWithLogitsLoss, self).__init__(size_average, reduce)
      self.register_buffer('weight', weight)
      self.register_buffer('pos_weight', pos_weight)
      self.label_smoothing = label_smoothing

    def forward(self, input, target):
      # See for explanation:
      # See for implementation: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/losses/losses_impl.py#L706
      if self.label_smoothing > 0:
        label_smoothing = self.label_smoothing
        num_classes = 2
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / num_classes
        target = target * smooth_positives + smooth_negatives

      return binary_cross_entropy_with_logits(input, target,
                                                self.weight,
                                                pos_weight=self.pos_weight,
                                                size_average=self.size_average,
                                                reduce=self.reduce)



# Copied and adapted from https://github.com/pytorch/pytorch/pull/3062/files

class RandomCycleIter:
  """Randomly iterate element in each cycle
  Example:
      >>> rand_cyc_iter = RandomCycleIter([1, 2, 3])
      >>> [next(rand_cyc_iter) for _ in range(10)]
      [2, 1, 3, 2, 3, 1, 1, 2, 3, 2]
  """

  def __init__(self, data):
    self.data_list = list(data)
    self.length = len(self.data_list)
    self.i = self.length - 1

  def __iter__(self):
    return self

  def __next__(self):
    self.i += 1
    if self.i == self.length:
      self.i = 0
      random.shuffle(self.data_list)
    return self.data_list[self.i]

  next = __next__  # Py2


def class_aware_sample_generator(cls_iter, data_iter_list, n):
  i = 0
  while i < n:
    yield next(data_iter_list[next(cls_iter)])
    i += 1

class ClassAwareSampler(Sampler):
  """Samples elements randomly, without replacement.
  Arguments:
      data_source (Dataset): dataset to sample from
  Implemented Class-Aware Sampling: https://arxiv.org/abs/1512.05830
  Li Shen, Zhouchen Lin, Qingming Huang, Relay Backpropagation for Effective
  Learning of Deep Convolutional Neural Networks, ECCV 2016
  By default num_samples equals to number of samples in the largest class
  multiplied by num of classes such that all samples can be sampled.
  """

  def __init__(self, data_source, num_samples=0, sample_over_classes=None):
    self.data_source = data_source
    n_cls = len(data_source.classes)
    if sample_over_classes is None:
      iter_over = RandomCycleIter(range(n_cls))
    else:
      iter_over = sample_over_classes
    self.class_iter = RandomCycleIter(iter_over)
    cls_data_list = [list() for _ in range(n_cls)]
    for i, (path, target, img_id) in enumerate(data_source.samples):
      for class_id in np.where(target > 0.5)[0]:
        cls_data_list[class_id].append(i)
    self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
    if num_samples:
      self.num_samples = num_samples
    else:
      self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)

  def __iter__(self):
    return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples)

  def __len__(self):
    return self.num_samples