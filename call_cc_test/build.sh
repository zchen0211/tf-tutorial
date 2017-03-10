TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# For Linux:
# g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2

# For Mac: Add "-undefined dynamic_lookup" for mac building
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2 -undefined dynamic_lookup
