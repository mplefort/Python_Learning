# fixes errors of failed to init CUBLANN and CUDNN from tensorflow after windows 10 recent update.
# see: https://github.com/tensorflow/tensorflow/issues/7072
# This init folder is called if any of the NNs are initialized from nn folder.

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)