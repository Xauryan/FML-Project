from dataclasses import dataclass
import random

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import (
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
DEFAULT_ACTIVATIONS = ("tanh", "relu", "sigmoid")
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class SplitData:
    images: np.ndarray
    labels: np.ndarray
    labels_cat: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    train: SplitData
    validation: SplitData
    test: SplitData


def set_random_seed(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lenet5(activation="tanh", use_maxpool=True, use_augmentation=True):
    """LeNet-5: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84) -> FC(10)."""
    Pool = MaxPooling2D if use_maxpool else AveragePooling2D

    layers = [Input(shape=INPUT_SHAPE)]

    if use_augmentation:
        layers += [
            RandomRotation(0.08, name="aug_rotate"),
            RandomTranslation(0.08, 0.08, name="aug_translate"),
            RandomZoom(0.08, name="aug_zoom"),
        ]

    layers += [
        Conv2D(6, (5, 5), activation=activation, name="conv1"),
        Pool((2, 2), strides=2, name="pool1"),
        Conv2D(16, (5, 5), activation=activation, name="conv2"),
        Pool((2, 2), strides=2, name="pool2"),
        Flatten(name="flatten"),
        Dropout(0.1, name="dropout1"),
        Dense(120, activation=activation, name="fc1"),
        Dropout(0.25, name="dropout2"),
        Dense(84, activation=activation, name="fc2"),
        Dense(NUM_CLASSES, activation="softmax", name="output"),
    ]

    return Sequential(layers)


def _prepare_split(images, labels):
    images = images.reshape((-1, *INPUT_SHAPE)).astype("float32") / 255.0
    labels = np.asarray(labels, dtype=np.int64)
    return SplitData(
        images=images,
        labels=labels,
        labels_cat=to_categorical(labels, NUM_CLASSES),
    )


def load_dataset(validation_size=0.1, random_state=DEFAULT_RANDOM_SEED):
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return DatasetBundle(
        train=_prepare_split(x_train, y_train),
        validation=_prepare_split(x_val, y_val),
        test=_prepare_split(x_test, y_test),
    )
