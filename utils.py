import time

import numpy as np

def vector_to_index_list(matrix):
  ret = []
  for vec in np.split(matrix, matrix.shape[0]):
    ret.append(list(np.where(vec[0])[0]))
  return ret

class Timer:
  def __init__(self, name=""):
    self.name = name

  def __enter__(self):
    self.start = time.clock()
    return self

  def __exit__(self, *args):
    self.end = time.clock()
    self.interval = self.end - self.start
    print('{} took {:.03} sec.'.format(self.name, self.interval))