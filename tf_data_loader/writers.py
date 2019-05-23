# -*- coding: utf-8 -*-
"""Graph dataset generator and loader

The `GraphDataset` stores the information about how to load and generate
datasets. To create a dataset, you need to specify the features that the
dataset will load. Here 'features' means an instance of a subclass of the
`MyFeature` class. As an example:

```
class ShortestPathGraphDataset(data_util.GraphDataset):
  def __init__(self, data_dir, min_nodes, max_nodes):
    feature_list = [
      data_util.GraphFeature(
          key='input_graph',
          node_feature_size=5,
          edge_feature_size=1,
          global_feature_size=1,
          dtype='float32',
          description='Graph to input to network'),
      data_util.GraphFeature(
          key='target_graph',
          node_feature_size=2,
          edge_feature_size=2,
          global_feature_size=1,
          dtype='float32',
          description='Graph to output from network'),
      # Example of a non-graph feature
      data_util.TensorFeature(
          key='adj_mat_dense',
          shape=[max_nodes, max_nodes],
          dtype='float32',
          description='Sparse adjacency matrix of input graph'),
    ]
    super(ShortestPathGraphDataset, self).__init__(data_dir, feature_list)
    self.min_nodes = min_nodes
    self.max_nodes = max_nodes
  def gen_sample(self, name):
    # ... Compute graphs and labels on graphs
    return {
        'input_graph': input_graph_dict,
        'target_graph': target_graph_dict,
        'adj_mat_dense': adj_mat_dense,
    }
```

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import datetime
import abc

import tqdm
import numpy as np
import tensorflow as tf
import yaml
import importlib

class DataWriter(abc.ABC):
  """Graph dataset writer and generator

  The `GraphDataset` stores the information about how to write and generate
  datasets.
  """
  MAX_IDX = 700 # TODO: Change this from MAX_IDX to MAX_TFRECORD_SIZE

  # TODO: Figure out how to handle feature dicts
  def __init__(self, data_dir, feature_list):
    """Initializes GraphDataset

    Args:
      data_dir: string giving the path to where the data is/will be stored
      feature_list: list of objects that are instances of subclesses of
        MyFeature, defining the objects in each element of the dataset
    """
    self.data_dir = data_dir
    self.sizes = {}
    self.features = {v.key: v for v in feature_list}

  @abc.abstractmethod
  def gen_sample(self, name, index):
    """Generate a sample for this dataset.

    This can either generate synthetically example or load appropriate data
    (e.g. images) to store into a tfrecord for fast loading.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      index: number identifer for this particular sample

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    return {name: index}

  def process_features(self, loaded_features):
    """Prepare features for storing into a tfrecord.

    This can either generate synthetically example or load appropriate data
    (e.g. images) to store into a tfrecord for fast loading.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    features = {}
    for key, feat in self.features.items():
      features.update(feat.get_feature_write(loaded_features[key]))
    return features

  # TODO: Make hooks to make this more general
  def convert_dataset(self, name, num_entries):
    """Writes out tfrecords using `gen_sample` for mode `name`

    This is the primary function for generating the dataset. It saves out the
    tfrecords in `os.path.join(self.data_dir, name)`. Displays a progress
    bar to show progress. Be careful with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    # Generate config file
    with open(os.path.join(self.data_dir,name,'config.yaml'),'w') as f:
      # TODO: Handle more general structures
      yaml_list = {}
      for key, value in self.features.items():
        d = value.to_yaml_dict()
        d['__name__'] = type(value).__name__
        yaml_list.append(d)
      yaml.dump(yaml_list, f)
    # Write out dataset
    self.sizes[name] = num_entries
    fname = '{:03d}.tfrecords'
    outfile = lambda idx: os.path.join(self.data_dir, name, fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))

    print('Writing dataset to {}/{}'.format(self.data_dir, name))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    for index in tqdm.tqdm(range(self.sizes[name])):
      # Generate new file if over limit
      if file_idx > self.MAX_IDX:
        file_idx = 0
        if writer:
          writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.gen_sample(name, index)
      features = self.process_features(loaded_features)
      example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(example.SerializeToString())
      file_idx += 1

    if writer:
      writer.close()
    # And save out a file with the creation time for versioning
    timestamp_file = '{}_timestamp.txt'.format(name)
    timestamp_str = 'TFrecord created {}'.format(datetime.datetime.now())
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      date_file.write(timestamp_str)

  def create_np_dataset(self, name, num_entries):
    """Writes out `npz` files using `gen_sample` for mode `name`

    This function generates the dataset in numpy form. This is in case you need
    to use placeholders. Displays a progress bar to show progress. Be careful
    with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    self.sizes[name] = num_entries
    fname = '{:04d}.npz'
    outfile = lambda idx: os.path.join(self.data_dir, name,
                                       fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))
    print('Writing dataset to {}'.format(os.path.join(self.data_dir,
                                                      name)))
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample(name, index)
      npz_dict = {}
      for key, feat in self.features.items():
        npz_dict.update(feat.npz_value(features[key]))
      np.savez(outfile(index), **npz_dict)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      contents = 'Numpy Dataset created {}'.format(datetime.datetime.now())
      date_file.write(contents)


# TODO: Maybe make gen_sample a passed in method??
class DataNpzWriter(abc.ABC):
  """Graph dataset writer and generator

  The `GraphDataset` stores the information about how to write and generate
  datasets.
  """
  MAX_IDX = 700 # TODO: Change this from MAX_IDX to MAX_TFRECORD_SIZE

  def __init__(self, data_dir, feature_list):
    """Initializes GraphDataset

    Args:
      data_dir: string giving the path to where the data is/will be stored
      feature_list: list of objects that are instances of subclesses of
        MyFeature, defining the objects in each element of the dataset
    """
    self.data_dir = data_dir
    self.sizes = {}
    self.features = {v.key: v for v in feature_list}

  @abc.abstractmethod
  def gen_sample(self, name, index):
    """Generate a sample for this dataset.

    This can either generate synthetically example or load appropriate data
    (e.g. images) to store into a tfrecord for fast loading.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      index: number identifer for this particular sample

    Returns:
      features: Dictionary of key strings to values of this sample
    """
    return {name: index}

  def create_np_dataset(self, name, num_entries):
    """Writes out `npz` files using `gen_sample` for mode `name`

    This function generates the dataset in numpy form. This is in case you need
    to use placeholders. Displays a progress bar to show progress. Be careful
    with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    self.sizes[name] = num_entries
    fname = '{:04d}.npz'
    outfile = lambda idx: os.path.join(self.data_dir, name,
                                       fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))
    print('Writing dataset to {}'.format(os.path.join(self.data_dir,
                                                      name)))
    for index in tqdm.tqdm(range(num_entries)):
      features = self.gen_sample(name, index)
      npz_dict = {}
      for key, feat in self.features.items():
        npz_dict.update(feat.npz_value(features[key]))
      np.savez(outfile(index), **npz_dict)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      contents = 'Numpy Dataset created {}'.format(datetime.datetime.now())
      date_file.write(contents)


