'''
Author: matiastang
Date: 2022-08-01 10:43:21
LastEditors: matiastang
LastEditTime: 2022-08-01 17:51:30
FilePath: /matias-TensorFlow/src/python/tensorflow_fashion_mnist.py
Description: 服装图像进行分类
'''
#!/usr/bin/python3
#coding=utf-8

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 标签
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# 查看数据
print(train_images.shape)
print(len(train_labels))

# 预处理数据
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
'显示'
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
'显示'
plt.show()

# 构建模型

## 设置层
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

## 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## 训练模型
model.fit(train_images, train_labels, epochs=10)

## 评估准确率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

'''
$ python3 /Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py
2.9.2
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
29515/29515 [==============================] - 0s 2us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26421880/26421880 [==============================] - 2s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
5148/5148 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4422102/4422102 [==============================] - 6s 1us/step
(60000, 28, 28)
60000
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 30701 (\N{CJK UNIFIED IDEOGRAPH-77ED}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 38772 (\N{CJK UNIFIED IDEOGRAPH-9774}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 24676 (\N{CJK UNIFIED IDEOGRAPH-6064}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 19978 (\N{CJK UNIFIED IDEOGRAPH-4E0A}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 34915 (\N{CJK UNIFIED IDEOGRAPH-8863}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 36830 (\N{CJK UNIFIED IDEOGRAPH-8FDE}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 35033 (\N{CJK UNIFIED IDEOGRAPH-88D9}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 22871 (\N{CJK UNIFIED IDEOGRAPH-5957}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 22836 (\N{CJK UNIFIED IDEOGRAPH-5934}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 34923 (\N{CJK UNIFIED IDEOGRAPH-886B}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 36816 (\N{CJK UNIFIED IDEOGRAPH-8FD0}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 21160 (\N{CJK UNIFIED IDEOGRAPH-52A8}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 38795 (\N{CJK UNIFIED IDEOGRAPH-978B}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 20937 (\N{CJK UNIFIED IDEOGRAPH-51C9}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 35044 (\N{CJK UNIFIED IDEOGRAPH-88E4}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 23376 (\N{CJK UNIFIED IDEOGRAPH-5B50}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 34924 (\N{CJK UNIFIED IDEOGRAPH-886C}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 22806 (\N{CJK UNIFIED IDEOGRAPH-5916}) missing from current font.
  plt.show()
/Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_fashion_mnist.py:53: UserWarning: Glyph 21253 (\N{CJK UNIFIED IDEOGRAPH-5305}) missing from current font.
  plt.show()
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-08-01 17:49:54.552963: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-08-01 17:49:54.553572: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-08-01 17:49:54.906544: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/10
2022-08-01 17:49:55.056764: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
1875/1875 [==============================] - 7s 3ms/step - loss: 0.5015 - accuracy: 0.8220
Epoch 2/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.3764 - accuracy: 0.8639
Epoch 3/10
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3416 - accuracy: 0.8752
Epoch 4/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.3152 - accuracy: 0.8844
Epoch 5/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2961 - accuracy: 0.8908
Epoch 6/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2819 - accuracy: 0.8964
Epoch 7/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2698 - accuracy: 0.8999
Epoch 8/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2580 - accuracy: 0.9042
Epoch 9/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2480 - accuracy: 0.9079
Epoch 10/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2404 - accuracy: 0.9097
2022-08-01 17:51:00.293389: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
313/313 - 1s - loss: 0.3376 - accuracy: 0.8855 - 903ms/epoch - 3ms/step

Test accuracy: 0.8855000138282776
'''