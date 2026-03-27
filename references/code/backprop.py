import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.utils import to_categorical

# 定义LeNet模型
def build_lenet(input_shape):
    model = Sequential()

    # Layer 1: Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', input_shape=input_shape))

    # Layer 2: Average Pooling
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    # Layer 3: Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))

    # Layer 4: Average Pooling
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    model.add(Flatten())

    # Layer 5: Fully Connected Layer
    model.add(Dense(units=120, activation='sigmoid'))

    # Layer 6: Fully Connected Layer
    model.add(Dense(units=84, activation='sigmoid'))

    # Output Layer
    model.add(Dense(units=10, activation='softmax'))

    return model

# 自定义回调函数以记录第一个特征图
class FeatureMapCallback(Callback):
    def __init__(self, layer_index):
        super(FeatureMapCallback, self).__init__()
        self.layer_index = layer_index
        self.feature_maps = []

    def on_epoch_end(self, epoch, logs=None):
        layer_output = self.model.layers[self.layer_index].output
        first_feature_map = layer_output[0, :, :, 0]
        self.feature_maps.append(first_feature_map)

# 加载MNIST数据集
(x_train, y_train), (_, _) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1)
y_train = to_categorical(y_train, 10)
input_shape = x_train.shape[1:]

# 构建LeNet模型
model = build_lenet(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 定义保存特征图的回调函数
callbacks = [FeatureMapCallback(layer_index=0)]  # 选择第一个卷积层

# 训练模型并记录特征图
model.fit(x_train, y_train, epochs=5, batch_size=32, callbacks=callbacks)

# 将特征图保存为视频
video_path = "feature_maps.mp4"
feature_maps = callbacks[0].feature_maps

# 创建视频编码器
# 创建视频编码器
fourcc = cv2.VideoWriter.fourcc(*'MP4V')  # 使用MP4V编码器
video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (feature_maps[0].shape[1], feature_maps[0].shape[0]))


# 将特征图添加到视频中
# 将特征图添加到视频中
for feature_map in feature_maps:
    # 将Keras张量转换为NumPy数组
    feature_map.eval(session=tf.compat.v1.keras.backend.get_session())

    # 对特征图进行处理
    feature_map_np = (feature_map_np * 255).astype(np.uint8)
    feature_map_np = cv2.cvtColor(feature_map_np, cv2.COLOR_GRAY2BGR)

    # 写入视频
    video_writer.write(feature_map_np)

# 释放视频编码器并完成
video_writer.release()
