'''
Author: matiastang
Date: 2022-07-26 16:33:46
LastEditors: matiastang
LastEditTime: 2022-08-01 17:24:29
FilePath: /matias-TensorFlow/src/python/tensorflow_mnist.py
Description: TensorFlow 2.0入门、mnist手写识别
'''
#!/usr/bin/python3
#coding=utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow

import tensorflow as tf

mnist = tf.keras.datasets.mnist

# 载入并准备好 MNIST 数据集。将样本从整数转换为浮点数：
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数：

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练并验证模型：

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

'''
$ python3 /Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_mnist.py
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 6s 0us/step
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-08-01 17:19:19.562493: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-08-01 17:19:19.562978: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-08-01 17:19:20.238354: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/5
2022-08-01 17:19:20.399801: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
1875/1875 [==============================] - 9s 4ms/step - loss: 0.2864 - accuracy: 0.9167
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1350 - accuracy: 0.9597
Epoch 3/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0986 - accuracy: 0.9703
Epoch 4/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0809 - accuracy: 0.9752
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0659 - accuracy: 0.9796
2022-08-01 17:19:57.527091: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
313/313 - 1s - loss: 0.0768 - accuracy: 0.9750 - 908ms/epoch - 3ms/step
'''