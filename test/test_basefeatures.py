from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import unittest

from tf_data_loader import basefeatures as basefeat

class NpDenseToSparseTest(unittest.TestCase):
  def test_reconstruct(self):
    A = np.array([[ 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                  [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                  [ 0.1, 0.0, 0.3, 0.0, 0.0, 0.0 ],
                  [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.1 ]])
    idx, vals = basefeat.np_dense_to_sparse(A)
    Arecon = np.zeros_like(A)
    Arecon[idx] = vals
    self.assertEqual(np.linalg.norm(Arecon-A), 0.0)

  def test_single_value(self):
    A = np.array([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                  [ 0.0, 0.0, 0.3, 0.0, 0.0, 0.0 ],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]])
    idx, vals = basefeat.np_dense_to_sparse(A)
    self.assertEqual(idx[0][0],2)
    self.assertEqual(idx[1][0],2)
    self.assertEqual(vals[0], 0.3)

  def test_multi_value(self):
    A = np.array([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ],
                  [ 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
                  [ 0.0, 0.0, 0.3, 0.0, 0.0, 0.0 ],
                  [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]])
    idx, vals = basefeat.np_dense_to_sparse(A)
    self.assertEqual(idx[0][0],1)
    self.assertEqual(idx[1][0],4)
    self.assertEqual(vals[0], 0.9)
    self.assertEqual(idx[0][1],2)
    self.assertEqual(idx[1][1],2)
    self.assertEqual(vals[1], 0.3)

if __name__ == "__main__":
  unittest.main()

# TODO: More granular unit tests on all the individual Features
# 
# 
# class TensorFeature(MyFeature):
#   """Class used for decoding tensors of fixed size."""
# 
#   def __init__(self, key, shape, dtype, description, **kwargs):
#     """Initialization of TensorFeature, giving specification of feature.
# 
#     Args:
#       key: string acting as name and identifier for this feature
#       shape: list/tuple of int values describing shape of this feature.
#       dtype: string for tf.dtype of this feature
#       description: string describing what this feature (for documentation)
#       kwargs: Any additional arguments
#     """
#     super(TensorFeature, self).__init__(key=key,
#                                         description=description,
#                                         shape=shape,
#                                         dtype=dtype)
# 
#   def get_feature_write(self, value):
#     v = value.astype(self.dtype).tobytes()
#     feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
#     return {self.key: feat}
# 
#   def get_feature_read(self):
#     return {self.key: tf.FixedLenFeature([], tf.string)}
# 
#   def tensors_to_item(self, keys_to_tensors):
#     tensor = keys_to_tensors[self.key]
#     tensor = tf.decode_raw(tensor, out_type=self.dtype)
#     tensor = tf.reshape(tensor, self.shape)
#     sess = tf.InteractiveSession()
#     return tensor
# 
# 
# class IntFeature(MyFeature):
#   """Class used for decoding a single serialized int64 value.
# 
#   This class is to store a single integer value e.g. the lengths of an array.
#   """
# 
#   def __init__(self, key, description, dtype='int64', **kwargs):
#     """Initialization of IntFeature, giving specification of feature.
# 
#     Args:
#       key: string acting as name and identifier for this feature
#       description: string describing what this feature (for documentation)
#       dtype: string for tf.dtype of this feature (either 'int64' or 'int32')
#         Default= 'int64'
#       kwargs: Any additional arguments
#     """
#     super(IntFeature, self).__init__(key=key,
#                                      description=description,
#                                      shape=[],
#                                      dtype='int64')
#     assert(dtype in ['int64', 'int32'])
#     self.convert_to = dtype
# 
#   def get_feature_write(self, value):
#     feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#     return {self.key: feat}
# 
#   def tensors_to_item(self, keys_to_tensors):
#     tensor = keys_to_tensors[self.key]
#     if self.convert_to != 'int64':
#       return tf.cast(tensor, dtype=self.convert_to)
#     else:
#       return tf.cast(tensor, dtype=tf.int64)
# 
#   def get_placeholder_and_feature(self, batch=True):
#     if batch:
#       placeholder = tf.placeholder(tf.int64, shape=[None])
#     else:
#       # placeholder = tf.placeholder(tf.int64, shape=())
#       placeholder = tf.placeholder(tf.int64, shape=[])
#     if self.convert_to != 'int64':
#       sample = tf.cast(placeholder, dtype=self.convert_to)
#     else:
#       sample = placeholder
#     return {self.key: placeholder}, sample
# 
# 
# class VarLenIntListFeature(MyFeature):
#   """Class used for decoding variable length int64 lists."""
# 
#   def __init__(self, key, dtype, description, **kwargs):
#     """Initialization of VarLenIntListFeature, giving specification of feature.
# 
#     Args:
#       key: string acting as name and identifier for this feature
#       dtype: string for tf.dtype of this feature
#       description: string describing what this feature (for documentation)
#       kwargs: Any additional arguments
#     """
#     super(VarLenIntListFeature, self).__init__(key=key,
#                                                description=description,
#                                                shape=[None],
#                                                dtype=dtype)
# 
#   def get_feature_write(self, value):
#     """Input `value` should be a list of integers."""
#     feat = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#     return {self.key: feat}
# 
#   def get_feature_read(self):
#     return {self.key: tf.VarLenFeature(tf.int64)}
# 
#   def tensors_to_item(self, keys_to_tensors):
#     tensor = keys_to_tensors[self.key]
#     tensor = tf.sparse_tensor_to_dense(tensor)
#     return tf.cast(tensor, self.dtype)
# 
#   def get_placeholder_and_feature(self, batch=True):
#     placeholder = tf.placeholder(self.dtype, shape=self.shape)
#     return {self.key: placeholder}, placeholder
# 
# 
# class VarLenFloatFeature(MyFeature):
#   """Class used for decoding variable shaped float tensors."""
# 
#   def __init__(self, key, shape, description, **kwargs):
#     """Initialization of VarLenIntListFeature, giving specification of feature.
# 
#     Args:
#       key: string acting as name and identifier for this feature
#       shape: list/tuple of int values describing shape of this feature.
#       description: string describing what this feature (for documentation)
#       kwargs: Any additional arguments
#     """
#     super(VarLenFloatFeature, self).__init__(key=key,
#                                              description=description,
#                                              shape=shape,
#                                              dtype='float32')
# 
#   def get_feature_write(self, value):
#     """Input `value` has to be compatible with this instance's shape."""
#     if isinstance(value, np.ndarray):
#       err_msg = "VarLenFloatFeature shape incompatible with input shape"
#       if len(value.shape) == len(self.shape):
#         for i, sz in enumerate(value.shape):
#           if self.shape[i] is not None:
#             assert sz == self.shape[i], err_msg
#       elif len(value.shape) == 1 and \
#             len(self.shape) == 2 and \
#             self.shape[0] is None:
#         assert value.shape[0] == self.shape[1], err_msg
#       else:
#         assert False, err_msg
#       flist = tf.train.FloatList(value=value.reshape(-1))
#     else:
#       flist = tf.train.FloatList(value=value)
#     return {self.key: tf.train.Feature(float_list=flist)}
# 
#   def get_feature_read(self):
#     return {self.key: tf.VarLenFeature(tf.float32)}
# 
#   def tensors_to_item(self, keys_to_tensors):
#     tensor = keys_to_tensors[self.key]
#     tensor = tf.sparse_tensor_to_dense(tensor)
#     shape = [s if s is not None else -1 for s in self.shape]
#     tensor = tf.reshape(tensor, shape)
#     return tensor
# 
#   def get_placeholder_and_feature(self, batch=True):
#     placeholder = tf.placeholder(self.dtype, shape=self.shape)
#     return {self.key: placeholder}, placeholder
# 
# 
# class SparseTensorFeature(MyFeature):
#   """Class used for decoding serialized sparse float tensors."""
# 
#   def __init__(self, key, shape, description, **kwargs):
#     """Initialization of SparseTensorFeature, giving specification of feature.
# 
#     Args:
#       key: string acting as name and identifier for this feature
#       shape: list/tuple of int values describing shape of this feature.
#       description: string describing what this feature (for documentation)
#     """
#     super(SparseTensorFeature, self).__init__(key=key,
#                                               description=description,
#                                               shape=shape,
#                                               dtype='float32')
# 
#   # TODO: Make these change into concatenating for 1 index tensor
#   def get_feature_write(self, value):
#     """Value should be a tuple `(idx, vals)`.
# 
#     Value should be a tuple `(idx, vals)` with `idx` being a tuple of lists of
#     `int` values of the same length and `vals` is a list of `self.dtype` values
#     the same length as `idx[0]`.
#     """
#     idx, vals = value[0], value[1]
#     sptensor_feature = {
#         '{}_{:02d}'.format(self.key, i):
#         tf.train.Feature(int64_list=tf.train.Int64List(value=idx[i]))
#         for i in range(len(self.shape))
#     }
#     sptensor_feature['{}_value'.format(self.key)] = \
#       tf.train.Feature(float_list=tf.train.FloatList(value=vals))
#     return sptensor_feature
# 
#   def get_feature_read(self):
#     feat_read = {
#         '{}_{:02d}'.format(self.key, i): tf.VarLenFeature(tf.int64)
#         for i in range(len(self.shape))
#     }
#     feat_read['{}_value'.format(self.key)] = tf.VarLenFeature(self.dtype)
#     return feat_read
# 
#   def tensors_to_item(self, keys_to_tensors):
#     indices_sp = [
#         keys_to_tensors['{}_{:02d}'.format(self.key, i)]
#         for i in range(len(self.shape))
#     ]
#     indices_list = [tf.sparse_tensor_to_dense(inds) for inds in indices_sp]
#     indices = tf.stack(indices_list, -1)
#     values_sp = keys_to_tensors['{}_value'.format(self.key)]
#     values = tf.sparse_tensor_to_dense(values_sp)
#     tensor = tf.SparseTensor(indices, values, self.shape)
#     return tensor
# 
#   def stack(self, arr):
#     concat_arr = [tf.sparse_reshape(x, [1] + self.shape) for x in arr]
#     return tf.sparse_concat(0, concat_arr)
# 
#   # Placeholder related
#   def get_placeholder_and_feature(self, batch):
#     placeholder = tf.sparse_placeholder(self.dtype)
#     return {self.key: placeholder}, placeholder
# 
#   def get_feed_dict(self, placeholders, values, batch):
#     idxs, vals = values[self.key + '_idx'], values[self.key + '_val']
#     if batch:
#       idxs = np.concatenate((np.zeros((len(idxs), 1)), idxs), -1)
#     val = tf.SparseTensorValue(idxs, vals, [1] + self.shape)
#     return {placeholders[self.key]: val}
# 
#   def npz_value(self, value):
#     idx_, val = value[0], value[1]
#     idx = np.stack(idx_, -1)
#     return {self.key + '_idx': idx, self.key + '_val': val}



