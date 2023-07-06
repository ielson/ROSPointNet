import numpy as np
import tensorflow as tf
import keras

from keras import layers


def conv_bn(x, filters):
    """Multilayer Perceptrons (MLPs) with shared weights"""
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    """Fully Connected (Linear/Dense) Layers"""
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    # the last layer shouldnt be relu, where is it passed?
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):
    # Initalize bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    # image description of PointNet t-net
    # https://miro.medium.com/v2/resize:fit:1400/format:webp/1*THOgzFqA6m4c8v-zSQCd6Q.png
    # MLPs values come from PointNet original paper
    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)

    x = layers.GlobalMaxPooling1D()(x)

    x = dense_bn(x, 512)
    # this is the last layer, the output is a vector of size num_features * num_features that will be transformed in a matrix of size num_features x num_features
    x = dense_bn(x, 256)


    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

