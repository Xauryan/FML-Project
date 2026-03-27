import PIL.Image
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['PATH'] += ';C:\\Program Files\\Graphviz\\bin\\'

# 加载保存的模型
model = load_model('lenet_mnist_trained_sigmoid.h5')

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 选择一张测试图像
# img = test_images[np.random.randint(0, test_images.shape[0])].reshape(1, 28, 28, 1)
img = test_images[3].reshape(1, 28, 28, 1)
# img = train_images[2500].reshape(1,28,28,1)
# 提取模型的前五层（卷积层和池化层）作为新的模型
feature_extractor = Model(inputs=model.inputs, outputs=[model.layers[0].output,
                                                        model.layers[1].output,
                                                        model.layers[2].output,
                                                        model.layers[3].output])

# 获取特征图
feature_maps1, feature_maps2, feature_maps3, feature_maps4 = feature_extractor.predict(img)
print(feature_maps3.shape)
# 可视化手写数字和前四个卷积层的特征图
plt.figure(figsize=(20, 8))

# 显示手写数字
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title('Input Image')
plt.axis('off')
plt.show()

# 显示前四个卷积层的特征图
# plt.subplot(1, 2, 2)
# 显示第一个卷积层的特征图
for i in range(5):  # 显示前5个特征图
    plt.subplot(4, 5, i+1)
    plt.imshow(feature_maps1[0, :, :, i], cmap='gray')
    # plt.title('Conv1 FM {}'.format(i+1))
    plt.axis('off')

# 显示第二个卷积层的特征图
for i in range(5):  # 显示前5个特征图
    plt.subplot(4, 5, i+6)
    plt.imshow(feature_maps2[0, :, :, i], cmap='gray')
    # plt.title('Conv2 FM {}'.format(i+1))
    plt.axis('off')

# 显示第三个卷积层的特征图
for i in range(5):  # 显示前5个特征图
    plt.subplot(4, 5, i+11)
    plt.imshow(feature_maps3[0, :, :, i], cmap='gray')
    # plt.title('Conv3 FM {}'.format(i+1))
    plt.axis('off')

# 显示第四个池化层的特征图
for i in range(5):  # 显示前5个特征图
    plt.subplot(4, 5, i+16)
    plt.imshow(feature_maps4[0, :, :, i], cmap='gray')
    # plt.title('Pool4 FM {}'.format(i+1))
    plt.axis('off')

plt.tight_layout()
plt.show()
