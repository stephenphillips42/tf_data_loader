# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import datetime

import tqdm
import numpy as np
import tensorflow as tf
import yaml
import importlib

class DataWriter(object):
  """Graph dataset writer

  The `GraphDataset` stores the information about how to write datasets.
  """
  MAX_IDX = 700 # TODO: Change this from MAX_IDX to MAX_TFRECORD_SIZE

  # TODO: Figure out how to handle feature dicts
  def __init__(self,
               data_dir,
               generator,
               feature_list=None,
               verbose=False):
    """Initializes GraphDataset

    Args:
      data_dir: string giving the path to where the data is/will be stored
      generator: DataGenerator object that generates the data samples
      feature_list: list of objects that are instances of subclesses of
        MyFeature, defining the objects in each element of the dataset.
        Default: None. Will use generator's feature_list if it is this
    """
    self.data_dir = data_dir
    self.sizes = {}
    self.generator = generator
    if feature_list is None:
      self.features = generator.features
    else:
      self.features = {v.key: v for v in feature_list}
    self.verbose = verbose
    # Generate config file
    if not os.path.exists(os.path.join(self.data_dir)):
      os.makedirs(os.path.join(self.data_dir))
    with open(os.path.join(self.data_dir,'config.yaml'),'w') as f:
      # TODO: Handle more general structures
      yaml_list = []
      for key, value in self.features.items():
        d = value.to_yaml_dict()
        d['__name__'] = type(value).__name__
        yaml_list.append(d)
      yaml.dump(yaml_list, f)

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
  def create_dataset(self, name, num_entries):
    """Writes out tfrecords using `gen_sample` for mode `name`

    This is the primary function for generating the dataset. It saves out the
    tfrecords in `os.path.join(self.data_dir, name)`. Displays a progress
    bar to show progress. Be careful with memory usage on disk.

    Args:
      name: name of the mode we are in (e.g. 'train', 'test')
      num_entries: number of entries (>= 1) to generate
    """
    # Write out dataset
    # TODO: Save out the sizes to the config file
    self.sizes[name] = num_entries
    fname = '{:03d}.tfrecords'
    outfile = lambda idx: os.path.join(self.data_dir, name, fname.format(idx))
    if not os.path.isdir(os.path.join(self.data_dir, name)):
      os.makedirs(os.path.join(self.data_dir, name))

    if self.verbose:
      print('Writing dataset to {}/{}'.format(self.data_dir, name))
    writer = None
    record_idx = 0
    file_idx = self.MAX_IDX + 1
    index_range = range(self.sizes[name])
    for index in tqdm.tqdm(index_range, disable=(not self.verbose)):
      # Generate new file if over limit
      if file_idx >= self.MAX_IDX:
        file_idx = 0
        if writer:
          writer.close()
        writer = tf.python_io.TFRecordWriter(outfile(record_idx))
        record_idx += 1
      loaded_features = self.generator.gen_sample(name, index)
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

# TODO: Maybe make gen_sample a passed in method??
class DataNpzWriter(DataWriter):
  """Graph dataset writer for npz files

  The `GraphNpzDataset` stores the information about how to write numpy
  datasets.
  """

  def create_dataset(self, name, num_entries):
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
    if self.verbose:
      print('Writing dataset to {}'.format(os.path.join(self.data_dir,
                                                        name)))
    for index in tqdm.tqdm(range(num_entries), disable=(not self.verbose)):
      features = self.generator.gen_sample(name, index)
      npz_dict = {}
      for key, feat in self.features.items():
        npz_dict.update(feat.npz_value(features[key]))
      np.savez(outfile(index), **npz_dict)

    # And save out a file with the creation time for versioning
    timestamp_file = 'np_test_timestamp.txt'
    with open(os.path.join(self.data_dir, timestamp_file), 'w') as date_file:
      contents = 'Numpy Dataset created {}'.format(datetime.datetime.now())
      date_file.write(contents)


