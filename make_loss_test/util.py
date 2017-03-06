import os
from timeit import default_timer as timer

import problems


def get_default_net_config(name, path):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": None
  }


def get_config(problem_name, path=None):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": None
    }}
    net_assignments = None
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  return problem, net_config, net_assignments
