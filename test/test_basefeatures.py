from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import unittest
import tempfile
import shutil

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

# Tests of reading and writing are done in test_writers_readers file
# These tests are for auxillary functions
# TFRecord tests
class TFRecordTest(unittest.TestCase):
  NTESTS = 4
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

  def tearDown(self):
    self.sess.close()
    # Remove the directory after each test
    shutil.rmtree(self.test_dir)


class TensorFeatureTFRecordTest(TFRecordTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_' + dtype
    np.random.seed(abs(hash(name)) % (2**31-1))
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
    for i in range(len(phs)):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X_[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack('int64')

  def test_stack_int32(self):
    self.type_test_stack('int32')

  def test_stack_float32(self):
    self.type_test_stack('float32')

  def test_stack_float64(self):
    self.type_test_stack('float64')


class IntFeatureTFRecordTest(TFRecordTest):
  def type_test_stack(self, dtype):
    name = 'test_stack_' + dtype.__name__
    np.random.seed(abs(hash(name)) % (2**31-1))
    feat = basefeat.IntFeature(key=name, 
                               dtype=dtype.__name__,
                               description='Feature ' + name)
    phs = [
      feat.get_placeholder_and_feature(batch=False)[1]
      for i in range(self.NTESTS)
    ]
    vals = [
      dtype(np.random.randint(-120,120))
      for i in range(self.NTESTS)
    ]
    X = feat.stack(phs)
    X_ = self.sess.run(X, feed_dict=dict(zip(phs,vals)))
    for i in range(len(phs)):
      msg = 'Value {} is not the same'.format(i)
      self.assertTrue(np.allclose(X_[i], vals[i]), msg)

  def test_stack_int64(self):
    self.type_test_stack(np.int64)

  def test_stack_int32(self):
    self.type_test_stack(np.int32)


class VarLenIntListFeatureTFRecordTest(TFRecordTest):
  pass


class VarLenFloatFeatureTFRecordTest(TFRecordTest):
  pass


class SparseTensorFeatureTFRecordTest(TFRecordTest):
  pass


if graph_nets_available:
  class GraphFeatureTFRecordTest(TFRecordTest):
    pass

# NPZ Tests
class TensorFeatureNPZTest(unittest.TestCase):
  pass

class IntFeatureNPZTest(unittest.TestCase):
  pass

class VarLenIntListFeatureNPZTest(unittest.TestCase):
  pass

class VarLenFloatFeatureNPZTest(unittest.TestCase):
  pass

class SparseTensorFeatureNPZTest(unittest.TestCase):
  pass

if graph_nets_available:
  class GraphFeatureNPZTest(unittest.TestCase):
    pass

if __name__ == "__main__":
  unittest.main()



