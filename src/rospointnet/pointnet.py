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
    print("tnet shape: {}".format(x))
    # TODO plot pointcloud rotated by tnet
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def backbone(num_points: int, num_classes: int, segmentation: bool = False):
    """
    This is the main part of the pointnet, before the classification or segmentation heads.
    It obtains the local and global features, that will be passed to the heads depending in what is asked

    """
    inputs = keras.Input(shape=(num_points, 3))
    print("shape of inputs: {}".format(inputs.shape))

    # input transform
    x = tnet(inputs, 3)

    # shared MLP 1
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    
    # feature transform
    x = tnet(x, 64)
    
    # store local features for segmentation head
    # not sure if this clone works
    local_features = tf.identity(x)
    
    
    # shared MLP 2
    x = conv_bn(x, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)

    # antigo comentado
    # x = layers.GlobalMaxPooling1D()(x)
    global_features = layers.GlobalMaxPooling1D()(x)
    # TODO return the skeleton (critical points) that comes from max pooling - are they the global features?
    print("Global features shape: {}".format(global_features.shape))
    #We refer to the global feature indices as the critical indices, since they index the points that are critical to the overall shape and structure of the point cloud.

    # codigo faz isso 
    # global_features = global_features.view(bs, -1)

    if segmentation:
        # features = torch.cat((local_features, global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), dim=1)
        return features, inputs # he alse returns the second t-net (feature transform)
    else:
        return global_features, inputs # he alse returns the second t-net (feature transform)


    """
    x = dense_bn(x, 512)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )
    
    return model
    """

def pointnetClassificationHead(num_points: int, num_classes: int):

    # first we need to get the backbone
    # self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

    backbone_output, inputs = backbone(num_points, num_classes, segmentation=False)

    x = backbone_output
    print("backbone output type: {}".format(type(x)))
    # MLP for classification
    x = dense_bn(x, 512)
    # why do we use a dropout here?
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 256)
    # in the other example theres just one dropout layer here
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model