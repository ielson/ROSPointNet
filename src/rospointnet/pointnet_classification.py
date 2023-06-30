import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from matplotlib import pyplot as plt

tf.random.set_seed(1234)

def set_dataset_directories() -> str:
    DATA_DIR = keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )
    DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")
    return DATA_DIR

def open_a_mesh_from_dataset(data_dir: str):
    mesh = trimesh.load(os.path.join(data_dir, "chair/train/chair_0001.off"))
    mesh.show()

    # TODO: Compare downsampling from Open3D with trimesh's sampling.
    points = mesh.sample(2048)


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_axis_off()
    plt.show()

def parse_dataset(data_dir: str,
                  num_points=2048):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
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

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def create_model(num_points: int, num_classes: int):
    """
    Creates a PointNet Model.
    """
    inputs = keras.Input(shape=(num_points, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
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

def test_trained_model():
    """
    Loads a model that has already been trained and tests it on a validation test set.
    """
    # Loads test data
    DATA_DIR = set_dataset_directories()
    NUM_POINTS = 2048
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    _, test_points, _, test_labels, CLASS_MAP = parse_dataset(DATA_DIR, NUM_POINTS)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    # Create a new model instance
    model = create_model(NUM_POINTS, NUM_CLASSES)

    model.load_weights('../pointnet_network_config/weights/modelnet10_weights.h5')

    # Tests model
    data = test_dataset.take(1)

    points, labels = list(data)[0]
    points = points[:8, ...]
    labels = labels[:8, ...]

    # Run test data through model
    preds = model.predict(points)
    preds = tf.math.argmax(preds, -1)

    points = points.numpy()

    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_title(
            "pred: {:}, label: {:}".format(
                CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
            )
        )
        ax.set_axis_off()
    plt.show()

def train_and_validate_model(epochs: int = 1,
        test_model: bool = True):
    
    # Checking if GPU is enabled
    print(f"\n\nGPU usage is set to : {tf.test.is_built_with_cuda()}.")

    DATA_DIR = set_dataset_directories()
    NUM_POINTS = 2048
    NUM_CLASSES = 10
    BATCH_SIZE = 32

    train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(
        DATA_DIR, NUM_POINTS
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    model = create_model(NUM_POINTS, NUM_CLASSES)

    # Trains model
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Saving the weights from training
    model.save_weights('../pointnet_network_config/weights/modelnet10_weights.h5')

    if test_model:
        data = test_dataset.take(1)

        points, labels = list(data)[0]
        points = points[:8, ...]
        labels = labels[:8, ...]

        # Run test data through model
        preds = model.predict(points)
        preds = tf.math.argmax(preds, -1)

        points = points.numpy()

        # plot points with predicted class and label
        fig = plt.figure(figsize=(15, 10))
        for i in range(8):
            ax = fig.add_subplot(2, 4, i + 1, projection="3d")
            ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
            ax.set_title(
                "pred: {:}, label: {:}".format(
                    CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
                )
            )
            ax.set_axis_off()
        plt.show()

if __name__ == "__main__":
    #train_and_validate_model()
    test_trained_model()

