from keras import backend as K
import numpy as np

def bilinear(shape, dtype=None):
  
  
  filter_size = shape[0]
  num_channels = shape[2]
  
  
  bilinear_kernel = np.zeros([filter_size, filter_size], dtype=dtype)
  scale_factor = (filter_size + 1) // 2
  if filter_size % 2 == 1:
    center = scale_factor - 1
  else:
    center = scale_factor - 0.5
  for x in range(filter_size):
    for y in range(filter_size):
      bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                               (1 - abs(y - center) / scale_factor)
  weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
  for i in range(num_channels):
    weights[:, :, i, i] = bilinear_kernel
  
  return weights
