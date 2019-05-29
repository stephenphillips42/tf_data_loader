# -*- coding: utf-8 -*-
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


class DataGenerator(abc.ABC):
  """Class to handle dataset generator for various `DatasetWriters`

  The class that stores the information about how to write and generate
  datasets.
  """

  # TODO: Figure out how to handle feature dicts
  def __init__(self, feature_list):
    """
    Args:
      feature_list: list of objects that are instances of subclesses of
        MyFeature, defining the objects in each element of the dataset
    """
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
      features: Dictionary of key strings to values of this sample. Should
        have the same keys as the MyFeature objects of feature_list
    """
    return { k: None for k, _ in self.features.items() }


