import numpy as np
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string("mode", "test", "train or test")
flags.DEFINE_string("model_name", "Auto_GAN", "Name of model")

# data paramters
# the data format is [num_sample, num_nodes, num_features,num_class]
flags.DEFINE_integer("num_samples", 2000, "number of samples")
flags.DEFINE_integer("num_nodes", 3, "number of nodes")
flags.DEFINE_integer("num_features", 24*4, "number of features")
flags.DEFINE_integer("num_class", 1, "number of classes to classify")
flags.DEFINE_integer("hid_layer1", 20, "number of first hidden layer for MLP")
flags.DEFINE_integer("hid_layer2", 20, "number of first hidden layer for MLP")
flags.DEFINE_integer("num_deep_feature", 20, "number of first hidden layer for MLP")
# training parameters
flags.DEFINE_integer("epoch", 2000, "number of epochs t train")
flags.DEFINE_float("learning_rate", 0.001, "learning rate of optimizer")
flags.DEFINE_float("momentum", 0.5, "momentum  of optimizer")
flags.DEFINE_integer("batch_size", 2, "batch size for training")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "directory name to save the checkpoints")

flags.DEFINE_float("alpha", 0.5, "alpha  of optimizer")
flags.DEFINE_float("beta", 0.5, "beta  of optimizer")

flags.DEFINE_boolean("use_autoencoder", False, "if autoencoder in network")
args = flags.FLAGS