from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import unittest
import tempfile
import shutil
import zlib


# Debug printing
import pprint
pp_xfawedfssa = pprint.PrettyPrinter(indent=2)
def myprint(x):
  if type(x) == str:
    print(x)
  else:
    pp_xfawedfssa.pprint(x)

# Graph related stuff
graph_nets_available = True
try:
  from graph_nets import graphs
  from graph_nets import utils_tf
  from graph_nets import utils_np
  import networkx as nx
except ImportError:
  print("Unable to use graph_nets")
  graph_nets_available = False

from tf_data_loader import basefeatures as basefeat

def myhash(x):
  return zlib.adler32(str(x).encode('utf-8'))

###### np_dense_to_sparse tests
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

###### Tests of reading and writing are done in test_writers_readers file
# These tests are for auxillary functions

# TFRecord tests - generic superclass
class TFRecordTest(unittest.TestCase):
  NTESTS = 4
  def setUp(self):
    # # Create a temporary directory for data saving before each test
    # self.test_dir = tempfile.mkdtemp()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

  def tearDown(self):
    self.sess.close()
    # # Remove the directory after each test
    # shutil.rmtree(self.test_dir)


class TensorFeatureTFRecordTest(TFRecordTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_'  + self.__class__.__name__ + '_' + dtype
    np.random.seed(abs(myhash(name)) % (2**31-1))
    shape = [3,4,3]
    feat = basefeat.TensorFeature(key=name, 
                                  shape=shape,
                                  dtype=dtype,
                                  description='Feature ' + name)
    phs = [
      feat.get_placeholder_and_feature(batch=False)[1]
      for i in range(self.NTESTS)
    ]
    vals = [
      np.random.randn(*shape).astype(dtype)
      for i in range(self.NTESTS)
    ]
    X = feat.stack(phs)
    X_ = self.sess.run(X, feed_dict=dict(zip(phs,vals)))
    for i in range(self.NTESTS):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X_[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack('int64')

  def test_stack_int32(self):
    self.type_test_stack('int32')

  def test_stack_int32(self):
    self.type_test_stack('uint8')

  def test_stack_float32(self):
    self.type_test_stack('float32')

  def test_stack_float64(self):
    self.type_test_stack('float64')


class IntFeatureTFRecordTest(TFRecordTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_'  + self.__class__.__name__ + '_' + dtype
    np.random.seed(abs(myhash(name)) % (2**31-1))
    feat = basefeat.IntFeature(key=name, 
                               dtype=dtype,
                               description='Feature ' + name)
    phs = [
      feat.get_placeholder_and_feature(batch=False)[1]
      for i in range(self.NTESTS)
    ]
    dtype_ = np.__dict__[dtype]
    vals = [
      dtype_(np.random.randint(-120,120))
      for i in range(self.NTESTS)
    ]
    X = feat.stack(phs)
    X_ = self.sess.run(X, feed_dict=dict(zip(phs,vals)))
    for i in range(self.NTESTS):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X_[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack('int64')

  def test_stack_int32(self):
    self.type_test_stack('int32')


class VarLenFeatureTFRecordTest(TFRecordTest):
  def shape_test_stack(self, feat, max_lens):
    np.random.seed(abs(myhash(feat.key)) % (2**31-1))
    num_none = len([ x for x in feat.shape if x is None ])
    self.assertEqual(num_none, len(max_lens), 'Test faulty')
    vardims = [ i for i, d in enumerate(feat.shape) if d is None ]
    phs = [
      feat.get_placeholder_and_feature(batch=False)[1]
      for i in range(self.NTESTS)
    ]
    def make_shape():
      new_shape = [ d for d in feat.shape ]
      for i, m in zip(vardims, max_lens):
        new_shape[i] = np.random.randint(2,m)
      return new_shape
    vals = [
      np.random.randn(*make_shape()).astype(feat.dtype)
      for i in range(self.NTESTS)
    ]
    X = feat.stack(phs)
    X_ = self.sess.run(X, feed_dict=dict(zip(phs,vals)))
    for i in range(self.NTESTS):
      # Agrees on overlap
      msg = 'Value {} is not the same'.format(i)
      indices = [  slice(None) for x in feat.shape ]
      for d in vardims:
        indices[d] = slice(0, vals[i].shape[d])
      X_test = X_[i][tuple(indices)]
      self.assertTrue(np.allclose(X_test, vals[i]), msg)
      # Zero elsewhere
      msg = 'Padding on {} is not zero'.format(i)
      for i, m in zip(vardims, max_lens):
        indices[i] = slice(vals[i].shape[d], None)
      self.assertTrue(np.allclose(X_[i][tuple(indices)], 0), msg)


class VarLenIntListFeatureTFRecordTest(VarLenFeatureTFRecordTest):
  def test_stack_int64(self):
    name = 'test_stack_1dim_' + self.__class__.__name__
    feat = basefeat.VarLenIntListFeature(key=name, 
                                         dtype='int64',
                                         description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)

  def test_stack_int32(self):
    name = 'test_stack_2dim_' + self.__class__.__name__
    feat = basefeat.VarLenIntListFeature(key=name, 
                                         dtype='int32',
                                         description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)


class VarLenFloatFeatureTFRecordTest(VarLenFeatureTFRecordTest):
  def test_stack_1dim(self):
    name = 'test_stack_1dim_' + self.__class__.__name__
    feat = basefeat.VarLenFloatFeature(key=name, 
                                       shape=[3,None,3],
                                       description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)

  def test_stack_2dim(self):
    name = 'test_stack_2dim_' + self.__class__.__name__
    feat = basefeat.VarLenFloatFeature(key=name, 
                                       shape=[3,None,3,None],
                                       description='Feature ' + name)
    max_lens = [12, 5]
    self.shape_test_stack(feat, max_lens)


class SparseTensorFeatureTFRecordTest(TFRecordTest):
  def shape_test_stack(self, feat):
    np.random.seed(abs(myhash(feat.key)) % (2**31-1))
    phs = [
      feat.get_placeholder_and_feature(batch=False)[1]
      for i in range(self.NTESTS)
    ]
    vals_dense = [ np.random.binomial(1,0.1,size=feat.shape)
                   for i in range(self.NTESTS) ]
    vals_sparse = []
    for A in vals_dense:
      idx, vals = basefeat.np_dense_to_sparse(A)
      vals_sparse.append(tf.SparseTensorValue(np.stack(idx,-1), vals, feat.shape))
    X = feat.stack(phs)
    X_ = self.sess.run(X, feed_dict=dict(zip(phs,vals_sparse)))
    total_len = sum([ len(v.values) for v in vals_sparse])
    run_len = len(X_.values)
    msg = 'Differing number of values: {} vs {}'.format(run_len, total_len)
    self.assertEqual(run_len, total_len, msg)
    for i, v in enumerate(vals_sparse):
      for vval, vinds in zip(v.values, v.indices):
        value_found = False
        for xval, xinds in zip(X_.values, X_.indices):
          if xinds[0] == i and np.allclose(xinds[1:], vinds):
            msg = 'Values {} and {} not equal with inds {}'.format(vval, xval, xinds)
            self.assertEqual(xval, vval)
            value_found = True
        self.assertTrue(value_found, msg='Values {}, indices {} not found'.format(vval, vinds))

  def test_stack_shape1(self):
    name = 'test_stack_shape1' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[3,4,3],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)

  def test_stack_shape2(self):
    name = 'test_stack_shape2' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[1,12,8],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)

  def test_stack_shape3(self):
    name = 'test_stack_shape3' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[5,12,8,3],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)


if graph_nets_available:
  # TODO: The stacking feature is implemented in the graph_nets library - how to test? 
  class GraphFeatureTFRecordTest(TFRecordTest):
    pass

###### NPZ Tests
class NPZTest(unittest.TestCase):
  NTESTS = 4

class TensorFeatureNPZTest(NPZTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_'  + self.__class__.__name__ + '_' + dtype
    np.random.seed(abs(myhash(name)) % (2**31-1))
    shape = [3,4,3]
    feat = basefeat.TensorFeature(key=name, 
                                  shape=shape,
                                  dtype=dtype,
                                  description='Feature ' + name)
    vals = [
      np.random.randn(*shape).astype(dtype)
      for i in range(self.NTESTS)
    ]
    X = feat.np_stack(vals)
    for i in range(self.NTESTS):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack('int64')

  def test_stack_int32(self):
    self.type_test_stack('int32')

  def test_stack_int32(self):
    self.type_test_stack('uint8')

  def test_stack_float32(self):
    self.type_test_stack('float32')

  def test_stack_float64(self):
    self.type_test_stack('float64')


class IntFeatureNPZTest(NPZTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_'  + self.__class__.__name__ + '_' + dtype
    np.random.seed(abs(myhash(name)) % (2**31-1))
    feat = basefeat.IntFeature(key=name, 
                               dtype=dtype,
                               description='Feature ' + name)
    dtype_ = np.__dict__[dtype]
    vals = [
      dtype_(np.random.randint(-120,120))
      for i in range(self.NTESTS)
    ]
    X = feat.np_stack(vals)
    for i in range(self.NTESTS):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack('int64')

  def test_stack_int32(self):
    self.type_test_stack('int32')


class VarLenFeatureNPZTest(TFRecordTest):
  def shape_test_stack(self, feat, max_lens):
    np.random.seed(abs(myhash(feat.key)) % (2**31-1))
    num_none = len([ x for x in feat.shape if x is None ])
    self.assertEqual(num_none, len(max_lens), 'Test faulty')
    vardims = [ i for i, d in enumerate(feat.shape) if d is None ]
    def make_shape():
      new_shape = [ d for d in feat.shape ]
      for i, m in zip(vardims, max_lens):
        new_shape[i] = np.random.randint(2,m)
      return new_shape
    vals = [
      np.random.randn(*make_shape()).astype(feat.dtype)
      for i in range(self.NTESTS)
    ]
    X = feat.np_stack(vals)
    for i in range(self.NTESTS):
      # Agrees on overlap
      msg = 'Value {} is not the same'.format(i)
      indices = [  slice(None) for x in feat.shape ]
      for d in vardims:
        indices[d] = slice(0, vals[i].shape[d])
      X_test = X[i][tuple(indices)]
      self.assertTrue(np.allclose(X_test, vals[i]), msg)
      # Zero elsewhere
      msg = 'Padding on {} is not zero'.format(i)
      for i, m in zip(vardims, max_lens):
        indices[i] = slice(vals[i].shape[d], None)
      self.assertTrue(np.allclose(X[i][tuple(indices)], 0), msg)


class VarLenIntListFeatureNPZTest(VarLenFeatureNPZTest):
  def test_stack_int64(self):
    name = 'test_stack_1dim_' + self.__class__.__name__
    feat = basefeat.VarLenIntListFeature(key=name, 
                                         dtype='int64',
                                         description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)

  def test_stack_int32(self):
    name = 'test_stack_2dim_' + self.__class__.__name__
    feat = basefeat.VarLenIntListFeature(key=name, 
                                         dtype='int32',
                                         description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)

class VarLenFloatFeatureNPZTest(VarLenFeatureNPZTest):
  def test_stack_1dim(self):
    name = 'test_stack_1dim_' + self.__class__.__name__
    feat = basefeat.VarLenFloatFeature(key=name, 
                                       shape=[3,None,3],
                                       description='Feature ' + name)
    max_lens = [12]
    self.shape_test_stack(feat, max_lens)

  def test_stack_2dim(self):
    name = 'test_stack_2dim_' + self.__class__.__name__
    feat = basefeat.VarLenFloatFeature(key=name, 
                                       shape=[3,None,3,None],
                                       description='Feature ' + name)
    max_lens = [12, 5]
    self.shape_test_stack(feat, max_lens)


class SparseTensorFeatureNPZTest(NPZTest):
  def shape_test_stack(self, feat):
    np.random.seed(abs(myhash(feat.key)) % (2**31-1))
    vals_dense = [ np.random.binomial(1,0.1,size=feat.shape)
                   for i in range(self.NTESTS) ]
    vals_sparse = []
    for A in vals_dense:
      idx, vals = basefeat.np_dense_to_sparse(A)
      vals_sparse.append(tf.SparseTensorValue(np.stack(idx,-1), vals, feat.shape))
    X = feat.stack(phs)
    total_len = sum([ len(v.values) for v in vals_sparse])
    run_len = len(X_.values)
    msg = 'Differing number of values: {} vs {}'.format(run_len, total_len)
    self.assertEqual(run_len, total_len, msg)
    for i, v in enumerate(vals_sparse):
      for vval, vinds in zip(v.values, v.indices):
        value_found = False
        for xval, xinds in zip(X_.values, X_.indices):
          if xinds[0] == i and np.allclose(xinds[1:], vinds):
            msg = 'Values {} and {} not equal with inds {}'.format(vval, xval, xinds)
            self.assertEqual(xval, vval)
            value_found = True
        self.assertTrue(value_found, msg='Values {}, indices {} not found'.format(vval, vinds))

  def test_stack_shape1(self):
    name = 'test_stack_shape1' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[3,4,3],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)

  def test_stack_shape2(self):
    name = 'test_stack_shape2' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[1,12,8],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)

  def test_stack_shape3(self):
    name = 'test_stack_shape3' + self.__class__.__name__
    feat = basefeat.SparseTensorFeature(key=name, 
                                        shape=[5,12,8,3],
                                        description='Feature ' + name)
    self.shape_test_stack(feat)


if graph_nets_available:
  # TODO: The stacking feature is implemented in the graph_nets library - how to test? 
  class GraphFeatureNPZTest(unittest.TestCase):
    pass

if __name__ == "__main__":
  unittest.main()



