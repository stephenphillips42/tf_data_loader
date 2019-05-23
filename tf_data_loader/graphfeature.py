from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from graph_nets import graphs
from graph_nets import utils_tf

from .myfeatures import *

GRAPH_KEYS = [
    'n_node', 'nodes', 'n_edge', 'edges', 'receivers', 'senders', 'globals'
]

class GraphFeature(MyFeature):
  """Custom class used for decoding serialized GraphsTuples."""

  def __init__(self, key, node_feature_size, edge_feature_size,
               global_feature_size, dtype, description, **kwargs):
    super(GraphFeature, self).__init__(key=key,
                                       description=description,
                                       shape=[],
                                       dtype=dtype)
    features_list = [
        IntFeature(
            key='n_node',
            dtype='int32',
            description='Number of nodes we are using'),
        VarLenFloatFeature(
            key='nodes',
            shape=[None, node_feature_size],
            description='Initial node embeddings for optimization'),
        IntFeature(
            key='n_edge',
            dtype='int32',
            description='Number of edges in this graph'),
        TensorFeature(
            key='globals',
            shape=[global_feature_size],
            dtype='float32',
            description='Edge features'),
        VarLenFloatFeature(
            key='edges',
            shape=[None, edge_feature_size],
            description='Edge features'),
        VarLenIntListFeature(
            key='receivers',
            dtype='int32',
            description='Recieving nodes for edges'),
        VarLenIntListFeature(
            key='senders',
            dtype='int32',
            description='Sending nodes for edges'),
    ]
    self.features = {}
    for feat in features_list:
      key = feat.key
      self.features[key] = feat
      self.features[key].key = '{}_{}'.format(self.key, key)
    self.node_feature_size = node_feature_size
    self.edge_feature_size = edge_feature_size
    self.global_feature_size = global_feature_size

  def get_feature_write(self, value):
    """Input `value` should be a dictionary for a `graph_net.GraphsTuple`

    Input `value` should be a dictionary generated by one of the methods in
    `graph_net.util_tf`, a data dictionary for the graph
    """
    feat_write = {}
    for key, feat in self.features.items():
      feat_write.update(feat.get_feature_write(value[key]))
    return feat_write

  def get_feature_read(self):
    feat_read = {}
    for _, feat in self.features.items():
      feat_read.update(feat.get_feature_read())
    return feat_read

  def tensors_to_item(self, keys_to_tensors):
    graph_dict = {}
    for key, feat in self.features.items():
      graph_dict[key] = feat.tensors_to_item(keys_to_tensors)
    # return graphs.GraphsTuple(**graph_dict)
    return graph_dict

  def stack(self, arr):
    return utils_tf.data_dicts_to_graphs_tuple(arr)

  # Placeholder related
  def get_placeholder_and_feature(self, batch):
    del batch # We do not need batch
    placeholders = {}
    sample_dict = {}
    for key, feat in self.features.items():
      # Due to how graphs are concatenated we only need batch dimension for
      # globals, n_node, and n_edge
      batch = False
      if key in ['globals', 'n_node', 'n_edge']:
        batch = True
      ph, val = feat.get_placeholder_and_feature(batch=batch)
      placeholders.update(ph)
      sample_dict[key] = val
    sample = graphs.GraphsTuple(**sample_dict)
    return placeholders, sample

  def get_feed_dict(self, placeholders, values, batch):
    fdict = {}
    for key, feat in self.features.items():
      # Due to how graphs are concatenated we only need batch dimension for
      # globals, n_node, and n_edge
      batch = False
      if key in ['globals', 'n_node', 'n_edge']:
        batch = True
      fdict.update(feat.get_feed_dict(placeholders, values, batch))
    return fdict

  def npz_value(self, values):
    graph_dict = {}
    for key, feat in self.features.items():
      graph_dict.update(feat.npz_value(values[key]))
    return graph_dict

  # Configuration saving and loading
  def to_yaml_dict(self):
    return {
      'key': self.key,
      'description': self.description,
      'shape': self.shape,
      'dtype': self.dtype,
      'node_feature_size': self.node_feature_size,
      'edge_feature_size': self.edge_feature_size,
      'global_feature_size': self.global_feature_size,
    }


