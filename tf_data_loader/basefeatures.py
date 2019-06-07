from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

def np_dense_to_sparse(arr):
  """Takes in a np.ndarray and returns a sparisification of it"""
  idx = np.where(arr != 0.0)
  return idx, arr[idx]

# TODO: Make one for storing images in compressed jpg or png format
class BaseFeature(object):
  """Class for decoding a serialized values in tfrecords or npz files.

  Base class that all other features should subclass. Handles writing out to
  tfrecords, as well as reading from them, and handling of placeholders and
  feed_dicts as well. Could be a single simple thing (e.g. a fixed size Tensor)
  or something more complicated (e.g. a GraphTuple)
  """

  def __init__(self, key, description, shape=None, dtype='float32', **kwargs):
    """Initialization of BaseFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      description: string describing what this feature (for documentation)
      shape: list/tuple of int values describing shape of this feature, if
        applicable. Default= None
      dtype: string for tf.dtype of this feature
      kwargs: Any additional arguments
    """
    super(BaseFeature, self).__init__()
    self.key = key
    self.description = description
    self.shape = shape if shape is not None else []
    self.dtype = dtype

  def get_feature_write(self, value):
    """Returns dictionary of things to write to a tfrecord.

    This creates a dictionary of items to write to the tfrecord. Typically to
    write to a tfrecord it needs to be broken down into basic Tensors (either
    fixed or variable length), and thus if the feature is more complicated this
    function breaks it down into its basic writable components, e.g.  indices
    and values of a sparse tensor.

    Args:
      value: value (usually np.ndarray, but could be anything) to write out to
        the tfrecord

    Returns:
      feat_write: Dictionary of key strings to np.ndarrays/lists to write to
        tfrecord
    """
    return {self.key: value}

  def get_feature_read(self):
    """Returns dictionary of things to read from a tfrecord.

    This creates a dictionary of items to read for the tfrecord parser. The
    keys of the dictionary should be used in `tensors_to_item` to combine into
    the usable form of the feature, e.g. indices and values of a sparse
    tensor.

    Returns:
      feat_read: Dictionary of key strings to values to read from a tfrecord
        (e.g. dictionary of tf.FixedLenFeature, tf.VarLenFeature, etc.)
    """
    return {self.key: tf.FixedLenFeature([], self.dtype)}

  def tensors_to_item(self, keys_to_tensors):
    """Collects relevant items together to create feature.

    This is for the case where there needs to be some post-processing of the
    features or combining of several sub-features to make the feature readable,
    e.g. getting indices and values for a sparse matrix feature. Final
    processing will be done in stack for the batching operation.

    Args:
      keys_to_tensors: dictionary of values loaded from the

    Returns:
      item: Combined values to create the final feature
    """
    item = keys_to_tensors[self.key]
    return item

  def stack(self, arr):
    """Stacks a list of parsed features for batching.

    This is called after loading features and calling tensors_to_item. It takes
    the values and concatenates them in an appropriate way so that the network
    can work with minibatches.

    Args:
      arr: list of parsed features to stack together

    Returns:
      item: item of values stacked together appropriately
    """
    return tf.stack(arr)

  # Placeholder related stuff
  def get_placeholder_and_feature(self, batch=True):
    """Gets dictionary of placeholders and computed value for this feature.

    In the case you are not using tfrecords, this can be used to create the
    appropriate placeholders for this feature. If this feature combines several
    basic components (e.g. a sparse tensor with indices and values) then it
    combines them together into a single value for this feature. Also handles
    batching of values within the placeholder.

    Args:
      batch: (bool, default=True) Whether to batch the output

    Returns:
      placeholder: Dictionary of key strings to placeholders for this feature
    """
    if batch:
      placeholder = tf.placeholder(self.dtype, shape=[None] + self.shape)
    else:
      placeholder = tf.placeholder(self.dtype, shape=self.shape)
    return {self.key: placeholder}, placeholder

  def get_feed_dict(self, placeholders, value_dict, batch=True):
    """Get the `feed_dict` for this feature, mapping placeholders to values.

    This creates the `feed_dict` by mapping the appropriate placeholders to the
    values provided. Also handles batching of values within the placeholder.

    Args:
      placeholders: Dictionary of key strings to placeholders
      value_dict: Dictionary of keys to values (typically np.ndarrays or lists)
        needed to build this feature
      batch: (bool, default=True) Whether to batch the output

    Returns:
      feed_dict: Dictionary of placeholders to values for this feature
    """
    if batch:
      val = np.expand_dims(value_dict[self.key], 0)  # Add batch dimension
    else:
      val = value_dict[self.key]
    return {placeholders[self.key]: val}

  def npz_value(self, value):
    return {self.key: value}

  def np_stack(self, arr):
    """Stacks a list of parsed features from the npz files for batching.

    This is called after loading features from the .npz file, to stack them
    into a batch.

    Args:
      arr: list of parsed features to stack together

    Returns:
      item: item of values stacked together appropriately
    """
    return np.stack(arr, axis=0)

  # Configuration saving and loading
  @classmethod
  def from_yaml_dict(cls, yaml_dict):
    return cls(**yaml_dict)

  def to_yaml_dict(self):
    return {
      'key': self.key,
      'description': self.description,
      'shape': self.shape,
      'dtype': self.dtype,
    }

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False
    if self.key != other.key:
      return False
    if self.description != other.description:
      return False
    if self.shape != other.shape:
      return False
    if self.dtype != other.dtype:
      return False
    return True


class TensorFeature(BaseFeature):
  """Class used for decoding tensors of fixed size."""

  def __init__(self, key, shape, dtype, description, **kwargs):
    """Initialization of TensorFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature.
      dtype: string for tf.dtype of this feature
      description: string describing what this feature (for documentation)
      kwargs: Any additional arguments
    """
    super(TensorFeature, self).__init__(key=key,
                                        description=description,
                                        shape=shape,
                                        dtype=dtype)

  def get_feature_write(self, value):
    v = value.astype(self.dtype).tobytes()
    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    return {self.key: feat}

  def get_feature_read(self):
    return {self.key: tf.FixedLenFeature([], tf.string)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.decode_raw(tensor, out_type=self.dtype)
    tensor = tf.reshape(tensor, self.shape)
    sess = tf.InteractiveSession()
    return tensor


class IntFeature(BaseFeature):
  """Class used for decoding a single serialized int64 value.

  This class is to store a single integer value e.g. the lengths of an array.
  """

  def __init__(self, key, description, dtype='int64', **kwargs):
    """Initialization of IntFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      description: string describing what this feature (for documentation)
      dtype: string for tf.dtype of this feature (either 'int64' or 'int32')
        Default= 'int64'
      kwargs: Any additional arguments
    """
    super(IntFeature, self).__init__(key=key,
                                     description=description,
                                     shape=[],
                                     dtype='int64')
    assert(dtype in ['int64', 'int32'])
    self.convert_to = dtype

  def get_feature_write(self, value):
    feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return {self.key: feat}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    if self.convert_to != 'int64':
      return tf.cast(tensor, dtype=self.convert_to)
    else:
      return tf.cast(tensor, dtype=tf.int64)

  def get_placeholder_and_feature(self, batch=True):
    if batch:
      placeholder = tf.placeholder(tf.int64, shape=[None])
    else:
      # placeholder = tf.placeholder(tf.int64, shape=())
      placeholder = tf.placeholder(tf.int64, shape=[])
    if self.convert_to != 'int64':
      sample = tf.cast(placeholder, dtype=self.convert_to)
    else:
      sample = placeholder
    return {self.key: placeholder}, sample

class VarLenTensorFeature(BaseFeature):
  """Class used for decoding variable size features."""

  def __init__(self, key, description, shape, dtype, **kwargs):
    """Initialization of VarLenTensorFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      description: string describing what this feature (for documentation)
      shape: list/tuple of int values describing shape of this feature, if
        applicable. Default= None
      dtype: string for tf.dtype of this feature
      kwargs: Any additional arguments
    """
    super(VarLenTensorFeature, self).__init__(key=key,
                                        description=description,
                                        shape=shape,
                                        dtype=dtype)
    self.pad_dims = [i for i, x in enumerate(self.shape) if x is None or x < 0]

  # Stacking functions
  def _pad(self, t, max_in_dims, constant_values=0):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    padt = tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
    return padt

  def stack(self, arr):
    # Much simpler code in the size 1 batch case
    if len(arr) == 1:
      return tf.stack(arr) 
    max_in_dims = [ d for d in self.shape ]
    for d in self.pad_dims:
      max_in_dims[d] = tf.reduce_max([ tf.shape(x)[d] for x in arr ])
    return tf.stack([ self._pad(x, max_in_dims) for x in arr ])

  def _pad_np(self, t, max_in_dims, constant_values=0):
    s = t.shape
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    padt = np.pad(t, paddings, 'constant', constant_values=constant_values)
    return padt

  def np_stack(self, arr):
    max_in_dims = [ d for d in self.shape ]
    for d in self.pad_dims:
      max_in_dims[d] = np.max([ x.shape[d] for x in arr ])
    return np.stack([ self._pad_np(x, max_in_dims) for x in arr ])


class VarLenIntListFeature(VarLenTensorFeature):
  """Class used for decoding variable length int64 lists."""

  def __init__(self, key, description, dtype='int64', **kwargs):
    """Initialization of VarLenIntListFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      dtype: string for tf.dtype of this feature. Default: int64
      description: string describing what this feature (for documentation)
      kwargs: Any additional arguments
    """
    super(VarLenIntListFeature, self).__init__(key=key,
                                               description=description,
                                               shape=[None],
                                               dtype=dtype)
    self.pad_dims = [i for i, x in enumerate(self.shape) if x is None or x < 0]

  def get_feature_write(self, value):
    """Input `value` should be a list of integers."""
    feat = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return {self.key: feat}

  def get_feature_read(self):
    return {self.key: tf.VarLenFeature(tf.int64)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    return tf.cast(tensor, self.dtype)


class VarLenFloatFeature(VarLenTensorFeature):
  """Class used for decoding variable shaped float tensors."""

  def __init__(self, key, shape, description, **kwargs):
    """Initialization of VarLenIntListFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature, or
        `None` along the dimension(s) of variable length.
      description: string describing what this feature (for documentation)
      kwargs: Any additional arguments
    """
    super(VarLenFloatFeature, self).__init__(key=key,
                                             description=description,
                                             shape=shape,
                                             dtype='float32')
    self.pad_dims = [i for i, x in enumerate(self.shape) if x is None or x < 0]

  def get_feature_write(self, value):
    """Input `value` has to be compatible with this instance's shape."""
    if isinstance(value, np.ndarray):
      err_msg = "VarLenFloatFeature shape incompatible with input shape"
      if len(value.shape) == len(self.shape):
        for i, sz in enumerate(value.shape):
          if self.shape[i] is not None:
            assert sz == self.shape[i], err_msg
      elif len(value.shape) == 1 and len(self.shape) == 2 and \
            self.shape[0] is None:
        assert value.shape[0] == self.shape[1], err_msg
      else:
        assert False, err_msg
      flist = tf.train.FloatList(value=value.reshape(-1))
    else:
      flist = tf.train.FloatList(value=value)
    return {self.key: tf.train.Feature(float_list=flist)}

  def get_feature_read(self):
    return {self.key: tf.VarLenFeature(tf.float32)}

  def tensors_to_item(self, keys_to_tensors):
    tensor = keys_to_tensors[self.key]
    tensor = tf.sparse_tensor_to_dense(tensor)
    shape = [s if s is not None else -1 for s in self.shape]
    tensor = tf.reshape(tensor, shape)
    return tensor


class SparseTensorFeature(BaseFeature):
  """Class used for decoding serialized sparse float tensors (only float32)."""

  def __init__(self, key, shape, description, **kwargs):
    """Initialization of SparseTensorFeature, giving specification of feature.

    Args:
      key: string acting as name and identifier for this feature
      shape: list/tuple of int values describing shape of this feature.
      description: string describing what this feature (for documentation)
    """
    super(SparseTensorFeature, self).__init__(key=key,
                                              description=description,
                                              shape=shape,
                                              dtype='float32')

  # TODO: Make these change into concatenating for 1 index tensor
  def get_feature_write(self, value):
    """Value should be a tuple `(idx, vals)`.

    Value should be a tuple `(idx, vals)` with `idx` being a tuple of lists of
    `int` values of the same length and `vals` is a list of `self.dtype` values
    the same length as `idx[0]`. This is given automatically if you use 
    `np_dense_to_sparse`.
    """
    idx, vals = value[0], value[1]
    sptensor_feature = {
        '{}_{:02d}'.format(self.key, i):
        tf.train.Feature(int64_list=tf.train.Int64List(value=idx[i]))
        for i in range(len(self.shape))
    }
    sptensor_feature['{}_value'.format(self.key)] = \
      tf.train.Feature(float_list=tf.train.FloatList(value=vals))
    return sptensor_feature

  def get_feature_read(self):
    feat_read = {
        '{}_{:02d}'.format(self.key, i): tf.VarLenFeature(tf.int64)
        for i in range(len(self.shape))
    }
    feat_read['{}_value'.format(self.key)] = tf.VarLenFeature(self.dtype)
    return feat_read

  def tensors_to_item(self, keys_to_tensors):
    indices_sp = [
        keys_to_tensors['{}_{:02d}'.format(self.key, i)]
        for i in range(len(self.shape))
    ]
    indices_list = [tf.sparse_tensor_to_dense(inds) for inds in indices_sp]
    indices = tf.stack(indices_list, -1)
    values_sp = keys_to_tensors['{}_value'.format(self.key)]
    values = tf.sparse_tensor_to_dense(values_sp)
    tensor = tf.SparseTensor(indices, values, self.shape)
    return tensor

  def stack(self, arr):
    concat_arr = [tf.sparse_reshape(x, [1] + self.shape) for x in arr]
    return tf.sparse_concat(0, concat_arr)

  def np_stack(self, arr):
    # inds_concat = [ np.reshape((i,*x[0]), [1] + self.shape)
    ind_prep = lambda i, x: np.stack([ i*np.ones(len(x[0])), *x ], axis=-1)
    inds_concat = [ ind_prep(i, x[0]) for i, x in enumerate(arr) ]
    vals_concat = [ x for _, x in arr ]
    return np.concatenate(inds_concat), np.concatenate(vals_concat)

  # Placeholder related
  def get_placeholder_and_feature(self, batch):
    placeholder = tf.sparse_placeholder(self.dtype)
    return {self.key: placeholder}, placeholder

  def get_feed_dict(self, placeholders, values, batch):
    idxs, vals = values[self.key + '_idx'], values[self.key + '_val']
    if batch:
      idxs = np.concatenate((np.zeros((len(idxs), 1)), idxs), -1)
    val = tf.SparseTensorValue(idxs, vals, [1] + self.shape)
    return {placeholders[self.key]: val}

  def npz_value(self, value):
    idx_, val = value[0], value[1]
    idx = np.stack(idx_, -1)
    return {self.key + '_idx': idx, self.key + '_val': val}

