import numpy as np

def vector_to_index_list(matrix):
  ret = []
  for vec in np.split(matrix, matrix.shape[0]):
    ret.append(list(np.where(vec[0])[0]))
  return ret
