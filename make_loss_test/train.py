import os

import tensorflow as tf

import meta
import util

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("problem", "simple", "Type of problem.")


def main(_):
  # problem is a function handle
  # slow learner: y outside loop
  # fast learner: x inside loop
  problem, net_config, net_assignments = util.get_config('simple')

  # Only slow learner will be instantiated during this phase
  x, constants = meta._get_variables(problem)
  print "Optimizee variables"
  print [op.name for op in x]
  print "Problem variables"
  print [op.name for op in constants]

  # Fast learner will be instantiated during this phase
  fx = meta._make_with_custom_variables(problem, x)
  assert(fx is not None)

  # for var in tf.global_variables():
  #   print var.name

  # meta.meta_minimize(problem)

if __name__ == '__main__':
  tf.app.run()
