# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tempfile
import shutil
import zlib
import unittest
import copy

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

# Our module
import tf_data_loader as tfdataloader

def myhash(x):
  return zlib.adler32(str(x).encode('utf-8'))

class BasicDataGenerator(tfdataloader.DataGenerator):
  """Test dataset generator """
  def __init__(self):
    features_list = [
      tfdataloader.TensorFeature(
          key='tensor',
          shape=[3,3,3],
          description='A tensor feature',
          dtype='float32',
        ),
      tfdataloader.IntFeature(
          key='class',
          description='A integer feature',
        ),
    ]
    super(BasicDataGenerator, self).__init__(features_list)
    self.out_val = None

  def gen_sample(self, name, index):
    # Pick a random seed value
    seed = abs(index * myhash(name)) % (2**32 - 1)
    np.random.seed(seed)
    # Generate data
    class_val =  np.random.randint(0,3)
    tensor = np.random.randn(3,3,3) + class_val
    self.out_val = {
      'tensor': tensor.astype('float32'),
      'class': class_val,
    }
    return self.out_val
basic_generator = BasicDataGenerator()

class AllFeaturesDataGenerator(tfdataloader.DataGenerator):
  """Test dataset generator """
  def __init__(self):
    self.tensor_shape = [4, 4, 4]
    self.num_auxilary_feats = 16
    features_list = [
      tfdataloader.TensorFeature(
          key='tensor',
          shape=self.tensor_shape,
          description='A tensor feature',
          dtype='float32',
        ),
      tfdataloader.IntFeature(
          key='class',
          dtype='int64',
          description='A integer feature',
        ),
      tfdataloader.VarLenIntListFeature(
          key='aux_classes',
          dtype='int32',
          description='A variable length integer feature',
        ),
      tfdataloader.VarLenFloatFeature(
          key='aux_classes_feat',
          shape=[None, self.num_auxilary_feats],
          description='A variable length float feature',
        ),
      tfdataloader.SparseTensorFeature(
          key='outlier_labels',
          shape=self.tensor_shape,
          description='A sparse tensor feature',
        ),
    ]
    super(AllFeaturesDataGenerator, self).__init__(features_list)

  def gen_sample(self, name, index):
    # Pick a random seed value
    seed = abs(index * myhash(name)) % (2**32 - 1)
    np.random.seed(seed)
    tshape = self.tensor_shape
    ashape = self.num_auxilary_feats
    # Generate data
    class_val =  np.random.randint(0, 3)
    nc = np.random.randint(2, 6)
    aux_classes = np.random.randint(0, 2, nc)
    aux_classes_feat = np.random.randn(nc, ashape) + aux_classes.reshape(-1, 1)
    outliers = np.random.binomial(1, 1./16., tshape)
    while np.sum(outliers) > 1:
      outliers = np.random.binomial(1, 1./16., tshape)
    tensor = (np.random.randn(*tshape) + outliers) + class_val
    outlier_labels = tfdataloader.np_dense_to_sparse(outliers.astype('float32'))
    return {
      'tensor': tensor.astype('float32'),
      'class': class_val,
      'aux_classes': aux_classes,
      'aux_classes_feat': aux_classes_feat.astype('float32'),
      'outlier_labels': outlier_labels,
    }
all_features_generator = AllFeaturesDataGenerator()


if graph_nets_available:
  class GraphDataGenerator(tfdataloader.DataGenerator):
    """Test dataset generator """
    def __init__(self):
      self.max_nodes = 8
      self.edge_prob = 0.2
      self.node_feature_size=8
      self.edge_feature_size=2
      self.global_feature_size=4
      features_list = [
        tfdataloader.GraphFeature(
            key='output_graph',
            node_feature_size=self.node_feature_size,
            edge_feature_size=self.edge_feature_size,
            global_feature_size=self.global_feature_size,
            description='A graph feature',
            dtype='float32',
          ),
        tfdataloader.TensorFeature(
            key='adj_mat_dense',
            shape=[self.max_nodes, self.max_nodes],
            description='A tensor feature',
            dtype='float32',
          ),
        tfdataloader.SparseTensorFeature(
            key='adj_mat_sparse',
            shape=[self.max_nodes, self.max_nodes],
            description='A sparse tensor feature',
          ),
      ]
      super(GraphDataGenerator, self).__init__(features_list)
      self.out_val = None

    def gen_sample(self, name, index):
      seed = abs(index * myhash(name)) % (2**32 - 1)
      np.random.seed(seed)
      # Pose graph and related objects
      num_nodes = np.random.randint(2,self.max_nodes)
      # Generate Erdos Renyi Graph
      AdjMat = np.random.binomial(1, self.edge_prob, (num_nodes, num_nodes))
      # Build spart graph representation
      G_nx = nx.from_numpy_matrix(AdjMat, create_using=nx.DiGraph)
      node_attrs = { i : np.random.randn(
                          self.node_feature_size
                        ).astype(np.float32)
                     for i in range(len(G_nx)) }
      edges_attrs = { (i, j) : np.random.randn(
                                 self.edge_feature_size
                               ).astype(np.float32)
                      for (i, j) in G_nx.edges }
      nx.set_node_attributes(G_nx, node_attrs, 'features')
      nx.set_edge_attributes(G_nx, edges_attrs, 'features')
      G = utils_np.networkx_to_data_dict(G_nx)
      G['globals'] = np.random.randn(
                       self.global_feature_size
                     ).astype('float32')
      # Build dense
      adj_mat_dense = np.zeros((self.max_nodes, self.max_nodes))
      adj_mat_dense[:num_nodes, :num_nodes] = AdjMat
      out_dict = {
        'output_graph': G,
        'adj_mat_dense' : adj_mat_dense,
        'adj_mat_sparse' : tfdataloader.np_dense_to_sparse(adj_mat_dense),
      }
      return out_dict
  graph_generator = GraphDataGenerator()


# Test cases
class DataWriterBasicTest(unittest.TestCase):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                               basic_generator,
                                               verbose=False)

  def tearDown(self):
    # Remove the directory after each test
    shutil.rmtree(self.test_dir)

  def test_config_write(self):
    path = os.path.join(self.test_dir, 'config.yaml')
    self.assertTrue(os.path.exists(path))
    load_passed = True
    try:
      import yaml
      with open(path, 'r') as f:
        config = yaml.load(f)
    except Exception as e:
      load_passed = False
    self.assertTrue(load_passed)

  def test_basic_write(self):
    name = 'test_basic_write'
    self.data_writer.create_dataset(name, 1)
    path = os.path.join(self.test_dir, name, '000.tfrecords')
    self.assertTrue(os.path.exists(path))

  # TODO: Change this behavior? (https://stackoverflow.com/questions/52191167/optimal-size-of-a-tfrecord-file)
  def test_large_write(self):
    name = 'test_large_write'
    self.data_writer.create_dataset(name, self.data_writer.MAX_IDX+1)
    path0 = os.path.join(self.test_dir, name, '000.tfrecords')
    self.assertTrue(os.path.exists(path0))
    path1 = os.path.join(self.test_dir, name, '001.tfrecords')
    self.assertTrue(os.path.exists(path1))


class DataWriterAllFeaturesTest(DataWriterBasicTest):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                               all_features_generator,
                                               verbose=False)


if graph_nets_available:
  class DataWriterGraphTest(DataWriterBasicTest):
    def setUp(self):
      # Create a temporary directory for data saving before each test
      self.test_dir = tempfile.mkdtemp()
      self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                                 graph_generator,
                                                 verbose=False)


class DataNpzWriterBasicTest(unittest.TestCase):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.data_writer = tfdataloader.DataNpzWriter(self.test_dir,
                                                  basic_generator,
                                                  verbose=False)

  def tearDown(self):
    # Remove the directory after each test
    shutil.rmtree(self.test_dir)

  def test_config_write(self):
    path = os.path.join(self.test_dir, 'config.yaml')
    self.assertTrue(os.path.exists(path))
    load_passed = True
    try:
      import yaml
      with open(path, 'r') as f:
        config = yaml.load(f)
    except Exception as e:
      load_passed = False
    self.assertTrue(load_passed)

  def test_basic_write(self):
    name = 'test_basic_write'
    self.data_writer.create_dataset(name, 1)
    path = os.path.join(self.test_dir, name, '0000.npz')
    self.assertTrue(os.path.exists(path))

  def test_large_write(self):
    name = 'test_basic_write_large'
    num_npz = 100
    self.data_writer.create_dataset(name, num_npz)
    for i in range(num_npz):
      path = os.path.join(self.test_dir, name, '{:04d}.npz'.format(i))
      self.assertTrue(os.path.exists(path))


class DataNpzWriterAllFeaturesTest(DataNpzWriterBasicTest):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.data_writer = tfdataloader.DataNpzWriter(self.test_dir,
                                                  all_features_generator,
                                                  verbose=False)


if graph_nets_available:
  class DataNpzWriterGraphTest(DataNpzWriterBasicTest):
    def setUp(self):
      # Create a temporary directory for data saving before each test
      self.test_dir = tempfile.mkdtemp()
      self.data_writer = tfdataloader.DataNpzWriter(self.test_dir,
                                                    graph_generator,
                                                    verbose=False)


class ReadTestCase(unittest.TestCase):
  def sparseTensorValToTuple(self, sptensor):
    indices = [
      x.reshape(-1)
      for x in np.split(sptensor.indices, sptensor.indices.shape[-1], axis=-1)
    ]
    values = sptensor.values
    ttensor = (indices, values)
    return ttensor

  # Have to deal with sparse tensor types
  def isEqualSampleVals(self, val0, val1):
    # Convert SparseTensorValues to tuples
    if isinstance(val0, tf.SparseTensorValue):
      val0 = self.sparseTensorValToTuple(val0)
    if isinstance(val1, tf.SparseTensorValue):
      val1 = self.sparseTensorValToTuple(val1)
    # Now treat equally
    if type(val0) == tuple:
      indices0, indices1 = val0[0], val1[0]
      values0, values1 = val0[1], val1[1]
      if len(indices0) != (len(indices1) + 1) and \
          len(indices0) != len(indices1): # Batching
        return False
      if len(indices0) == (len(indices1) + 1):
        indices0 = indices0[1:]
      if len(indices0[0]) != len(indices1[0]):
        return False
      if values0.shape != values1.shape:
        return False
      for ind0, ind1 in zip(indices0, indices1):
        if not np.array_equiv(ind0,ind1):
          return False
      return np.array_equiv(values0, values1)
    else:
      equal_val = np.array_equiv(np.squeeze(val0), np.squeeze(val1))
    return equal_val


class DataReaderBasicTest(ReadTestCase):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.generator = basic_generator
    self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                               self.generator,
                                               verbose=False)
    self.data_reader = tfdataloader.DataReader(self.test_dir)
 
  def tearDown(self):
    # Remove the directory after each test
    shutil.rmtree(self.test_dir)

  def test_features(self):
    for key, write_feat in self.data_writer.features.items():
      read_feat = self.data_reader.features[key]
      self.assertEqual(read_feat, write_feat)

  def test_basic_read(self):
    name = 'test_basic_read'
    num_total = 1
    batch_size = 1
    self.data_writer.create_dataset(name, num_total)
    mysample = self.generator.gen_sample(name, 0)
    mysample2 = self.generator.gen_sample(name, 0)
    sample = self.data_reader.load_batch(name, batch_size, shuffle_data=False)
    with tf.Session(config=copy.deepcopy(self.config)) as sess:
      sample_ = sess.run(sample)
      self.assertEqual(sample_.keys(), mysample.keys())
      for k in sorted(list(sample_.keys())):
        # equal_val = self.isEqualSampleVals(sample_[k], mysample[k])
        equal_val = self.isEqualSampleVals(mysample[k], mysample2[k])
        self.assertTrue(equal_val, '{} not equal'.format(k))

  def test_multi_read(self):
    name = 'test_multi_read'
    num_total = 10
    batch_size = 1
    self.data_writer.create_dataset(name, num_total)
    sample = self.data_reader.load_batch(name, batch_size, shuffle_data=False)
    with tf.Session(config=copy.deepcopy(self.config)) as sess:
      for b in range(num_total):
        sample_ = sess.run(sample)
        mysample = self.generator.gen_sample(name, b)
        self.assertEqual(sample_.keys(), mysample.keys())
        for k in sorted(list(sample_.keys())):
          equal_val = self.isEqualSampleVals(sample_[k], mysample[k])
          self.assertTrue(equal_val, '{} not equal'.format(k))


class DataReaderAllFeaturesTest(DataReaderBasicTest):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True
    self.generator = all_features_generator
    self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                               self.generator,
                                               verbose=False)
    self.data_reader = tfdataloader.DataReader(self.test_dir)
 

if graph_nets_available:
  class DataReaderGraphTest(DataReaderBasicTest):
    def setUp(self):
      # Create a temporary directory for data saving before each test
      self.test_dir = tempfile.mkdtemp()
      self.config = tf.ConfigProto()
      self.config.gpu_options.allow_growth = True
      self.generator = graph_generator
      self.data_writer = tfdataloader.DataWriter(self.test_dir,
                                                 self.generator,
                                                 verbose=False)
      self.data_reader = tfdataloader.DataReader(self.test_dir)

    # Have to deal with sparse tensor types
    def isEqualSampleVals(self, val0, val1):
      if isinstance(val0, graphs.GraphsTuple) and isinstance(val1, dict):
        val0 = utils_np.graphs_tuple_to_data_dicts(val0)[0]
        val1 = utils_np.data_dicts_to_graphs_tuple([val1])
        val1 = utils_np.graphs_tuple_to_data_dicts(val1)[0]
      if type(val0) == type(val1) and isinstance(val0, dict):
        ret_val = True
        for k in val0.keys():
          if not self.isEqualSampleVals(val0[k], val1[k]):
            ret_val = False
        return ret_val
      else:
        return super(DataReaderGraphTest, self).isEqualSampleVals(val0,val1)


class DataNpzReaderBasicTest(ReadTestCase):
  def setUp(self):
    # Create a temporary directory for data saving before each test
    self.test_dir = tempfile.mkdtemp()
    self.generator = basic_generator
    self.data_writer = tfdataloader.DataNpzWriter(self.test_dir,
                                                  self.generator,
                                                  verbose=False)
    self.data_reader = tfdataloader.DataNpzReader(self.test_dir)

  def tearDown(self):
    # Remove the directory after each test
    shutil.rmtree(self.test_dir)

  def test_features(self):
    for key, write_feat in self.data_writer.features.items():
      read_feat = self.data_reader.features[key]
      self.assertEqual(read_feat, write_feat)

  def test_load_npz_read(self):
    name = 'test_load_npz_read'
    self.data_writer.create_dataset(name, 1)
    mysample = self.generator.gen_sample(name, 0)
    sample = self.data_reader.load_npz_file(name, 0)
    for k in sorted(list(sample.keys())):
      equal_val = self.isEqualSampleVals(sample[k], mysample[k])
      self.assertTrue(equal_val, '{} not equal'.format(k))

  def test_multi_load_npz_read(self):
    name = 'test_multi_load_npz_read'
    num_total = 10
    batch_size = 1
    self.data_writer.create_dataset(name, num_total)
    for b in range(num_total):
      sample_ = self.data_reader.load_npz_file(name, b)
      mysample = self.generator.gen_sample(name, b)
      self.assertEqual(sample_.keys(), mysample.keys())
      for k in sorted(list(sample_.keys())):
        equal_val = self.isEqualSampleVals(sample_[k], mysample[k])
        self.assertTrue(equal_val, '{} not equal'.format(k))

  # def test_multi_load_npz_read(self):
  #   name = 'test_multi_load_npz_read'
  #   num_total = 10
  #   batch_size = 1
  #   self.data_writer.create_dataset(name, num_total)
  #   sample = self.data_reader.load_batch(name, batch_size, shuffle_data=False)
  #   with tf.Session(config=copy.deepcopy(self.config)) as sess:
  #     for b in range(num_total):
  #       sample_ = sess.run(sample)
  #       mysample = self.generator.gen_sample(name, b)
  #       self.assertEqual(sample_.keys(), mysample.keys())
  #       for k in sorted(list(sample_.keys())):
  #         equal_val = self.isEqualSampleVals(sample_[k], mysample[k])
  #         self.assertTrue(equal_val, '{} not equal'.format(k))


