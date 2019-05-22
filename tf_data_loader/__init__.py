
from .data_util import *
graph_net_exists = True
try:
  from .graphfeature import GraphFeature
except ImportError:
  graph_net_exists = False


