# 任选一种经典的卷积神经网络，实现对手写数字的识别，并对其模型架构、滤波器及特征图进行可视化分析。
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def build_lenet(input_shape):
    model = Sequential()

    # Layer 1: Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape))

    # Layer 2: Average Pooling
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    # Layer 3: Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))

    # Layer 4: Average Pooling
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    model.add(Flatten())

    # Layer 5: Fully Connected Layer
    model.add(Dense(units=120, activation='tanh'))

    # Layer 6: Fully Connected Layer
    model.add(Dense(units=84, activation='tanh'))

    # Output Layer
    model.add(Dense(units=10, activation='softmax'))

    return model


# Input shape for MNIST dataset
input_shape = (28, 28, 1)
model = build_lenet(input_shape)

# Load and preprocess the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and reshape the images
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Compile and train the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Variables to store training history
history = model.fit(train_images, train_labels, epochs=10, batch_size=32,
                    validation_data=(test_images, test_labels), verbose=1)

# Save history data as .dat files
np.savetxt('loss.dat', history.history['loss'])
np.savetxt('val_loss.dat', history.history['val_loss'])
np.savetxt('accuracy.dat', history.history['accuracy'])
np.savetxt('val_accuracy.dat', history.history['val_accuracy'])

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
model.save('lenet_mnist.h5')
