'''
Author: matiastang
Date: 2022-08-02 10:29:59
LastEditors: matiastang
LastEditTime: 2022-08-02 11:21:19
FilePath: /matias-TensorFlow/src/python/tensorflow_imdb.py
Description: IMDB影评
'''
#!/usr/bin/python3
#coding=utf-8

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.version)

# 下载数据（IMDB 数据集已经打包在 Tensorflow 中。该数据集已经经过预处理，评论（单词序列）已经被转换为整数序列，其中每个整数表示字典中的特定单词。）
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 处理数据
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 构建模型

# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 训练
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建一个验证集

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# model.fit() 返回一个 History 对象，该对象包含一个字典，其中包含训练阶段所发生的一切事件
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型

results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)
# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# loss图

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 清除数字

# accuracy图

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 在该图中，点代表训练损失值（loss）与准确率（accuracy），实线代表验证损失值（loss）与准确率（accuracy）
'''
python3 tensorflow_imdb.py 
<module 'tensorflow._api.v2.version' from '/opt/homebrew/Caskroom/miniforge/base/envs/mt_tensorflow/lib/python3.10/site-packages/tensorflow/_api/v2/version/__init__.py'>
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 [==============================] - 1s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
1641221/1641221 [==============================] - 1s 0us/step
Metal device set to: Apple M1

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

2022-08-02 11:16:32.787990: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-08-02 11:16:32.788532: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 16)          160000    
                                                                 
 global_average_pooling1d (G  (None, 16)               0         
 lobalAveragePooling1D)                                          
                                                                 
 dense (Dense)               (None, 16)                272       
                                                                 
 dense_1 (Dense)             (None, 1)                 17        
                                                                 
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________
2022-08-02 11:16:33.070104: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
Epoch 1/40
2022-08-02 11:16:33.375209: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
30/30 [==============================] - ETA: 0s - loss: 0.6919 - accuracy: 0.51472022-08-02 11:16:40.491506: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
30/30 [==============================] - 8s 200ms/step - loss: 0.6919 - accuracy: 0.5147 - val_loss: 0.6900 - val_accuracy: 0.6160
Epoch 2/40
30/30 [==============================] - 6s 190ms/step - loss: 0.6859 - accuracy: 0.6203 - val_loss: 0.6820 - val_accuracy: 0.7190
Epoch 3/40
30/30 [==============================] - 6s 186ms/step - loss: 0.6740 - accuracy: 0.7384 - val_loss: 0.6672 - val_accuracy: 0.7323
Epoch 4/40
30/30 [==============================] - 6s 186ms/step - loss: 0.6530 - accuracy: 0.7593 - val_loss: 0.6433 - val_accuracy: 0.7622
Epoch 5/40
30/30 [==============================] - 5s 180ms/step - loss: 0.6221 - accuracy: 0.7845 - val_loss: 0.6107 - val_accuracy: 0.7826
Epoch 6/40
30/30 [==============================] - 5s 183ms/step - loss: 0.5821 - accuracy: 0.8097 - val_loss: 0.5710 - val_accuracy: 0.7971
Epoch 7/40
30/30 [==============================] - 5s 179ms/step - loss: 0.5368 - accuracy: 0.8269 - val_loss: 0.5289 - val_accuracy: 0.8175
Epoch 8/40
30/30 [==============================] - 5s 183ms/step - loss: 0.4904 - accuracy: 0.8448 - val_loss: 0.4884 - val_accuracy: 0.8308
Epoch 9/40
30/30 [==============================] - 6s 185ms/step - loss: 0.4470 - accuracy: 0.8587 - val_loss: 0.4517 - val_accuracy: 0.8428
Epoch 10/40
30/30 [==============================] - 5s 181ms/step - loss: 0.4082 - accuracy: 0.8704 - val_loss: 0.4203 - val_accuracy: 0.8493
Epoch 11/40
30/30 [==============================] - 5s 181ms/step - loss: 0.3747 - accuracy: 0.8798 - val_loss: 0.3942 - val_accuracy: 0.8573
Epoch 12/40
30/30 [==============================] - 5s 175ms/step - loss: 0.3465 - accuracy: 0.8865 - val_loss: 0.3736 - val_accuracy: 0.8622
Epoch 13/40
30/30 [==============================] - 5s 179ms/step - loss: 0.3226 - accuracy: 0.8925 - val_loss: 0.3569 - val_accuracy: 0.8652
Epoch 14/40
30/30 [==============================] - 6s 190ms/step - loss: 0.3021 - accuracy: 0.8978 - val_loss: 0.3439 - val_accuracy: 0.8678
Epoch 15/40
30/30 [==============================] - 5s 178ms/step - loss: 0.2840 - accuracy: 0.9029 - val_loss: 0.3313 - val_accuracy: 0.8748
Epoch 16/40
30/30 [==============================] - 5s 175ms/step - loss: 0.2683 - accuracy: 0.9087 - val_loss: 0.3226 - val_accuracy: 0.8766
Epoch 17/40
30/30 [==============================] - 6s 189ms/step - loss: 0.2548 - accuracy: 0.9113 - val_loss: 0.3143 - val_accuracy: 0.8787
Epoch 18/40
30/30 [==============================] - 6s 187ms/step - loss: 0.2416 - accuracy: 0.9177 - val_loss: 0.3080 - val_accuracy: 0.8800
Epoch 19/40
30/30 [==============================] - 6s 191ms/step - loss: 0.2297 - accuracy: 0.9219 - val_loss: 0.3031 - val_accuracy: 0.8803
Epoch 20/40
30/30 [==============================] - 6s 192ms/step - loss: 0.2193 - accuracy: 0.9256 - val_loss: 0.2992 - val_accuracy: 0.8797
Epoch 21/40
30/30 [==============================] - 6s 185ms/step - loss: 0.2097 - accuracy: 0.9283 - val_loss: 0.2950 - val_accuracy: 0.8825
Epoch 22/40
30/30 [==============================] - 5s 174ms/step - loss: 0.2002 - accuracy: 0.9319 - val_loss: 0.2923 - val_accuracy: 0.8836
Epoch 23/40
30/30 [==============================] - 6s 186ms/step - loss: 0.1919 - accuracy: 0.9357 - val_loss: 0.2899 - val_accuracy: 0.8834
Epoch 24/40
30/30 [==============================] - 5s 176ms/step - loss: 0.1838 - accuracy: 0.9397 - val_loss: 0.2880 - val_accuracy: 0.8841
Epoch 25/40
30/30 [==============================] - 6s 186ms/step - loss: 0.1763 - accuracy: 0.9426 - val_loss: 0.2868 - val_accuracy: 0.8849
Epoch 26/40
30/30 [==============================] - 6s 185ms/step - loss: 0.1690 - accuracy: 0.9468 - val_loss: 0.2858 - val_accuracy: 0.8857
Epoch 27/40
30/30 [==============================] - 6s 187ms/step - loss: 0.1625 - accuracy: 0.9493 - val_loss: 0.2862 - val_accuracy: 0.8847
Epoch 28/40
30/30 [==============================] - 6s 185ms/step - loss: 0.1561 - accuracy: 0.9512 - val_loss: 0.2854 - val_accuracy: 0.8852
Epoch 29/40
30/30 [==============================] - 6s 186ms/step - loss: 0.1501 - accuracy: 0.9543 - val_loss: 0.2868 - val_accuracy: 0.8844
Epoch 30/40
30/30 [==============================] - 5s 173ms/step - loss: 0.1446 - accuracy: 0.9562 - val_loss: 0.2864 - val_accuracy: 0.8857
Epoch 31/40
30/30 [==============================] - 5s 177ms/step - loss: 0.1393 - accuracy: 0.9585 - val_loss: 0.2876 - val_accuracy: 0.8861
Epoch 32/40
30/30 [==============================] - 6s 185ms/step - loss: 0.1337 - accuracy: 0.9610 - val_loss: 0.2883 - val_accuracy: 0.8861
Epoch 33/40
30/30 [==============================] - 6s 186ms/step - loss: 0.1287 - accuracy: 0.9628 - val_loss: 0.2901 - val_accuracy: 0.8850
Epoch 34/40
30/30 [==============================] - 6s 187ms/step - loss: 0.1241 - accuracy: 0.9650 - val_loss: 0.2917 - val_accuracy: 0.8859
Epoch 35/40
30/30 [==============================] - 6s 187ms/step - loss: 0.1197 - accuracy: 0.9670 - val_loss: 0.2935 - val_accuracy: 0.8848
Epoch 36/40
30/30 [==============================] - 6s 185ms/step - loss: 0.1151 - accuracy: 0.9683 - val_loss: 0.2955 - val_accuracy: 0.8848
Epoch 37/40
30/30 [==============================] - 6s 185ms/step - loss: 0.1110 - accuracy: 0.9696 - val_loss: 0.2978 - val_accuracy: 0.8835
Epoch 38/40
30/30 [==============================] - 6s 191ms/step - loss: 0.1070 - accuracy: 0.9713 - val_loss: 0.3001 - val_accuracy: 0.8838
Epoch 39/40
30/30 [==============================] - 6s 184ms/step - loss: 0.1033 - accuracy: 0.9721 - val_loss: 0.3025 - val_accuracy: 0.8836
Epoch 40/40
30/30 [==============================] - 6s 188ms/step - loss: 0.0998 - accuracy: 0.9735 - val_loss: 0.3048 - val_accuracy: 0.8827
782/782 - 2s - loss: 0.3251 - accuracy: 0.8727 - 2s/epoch - 3ms/step
[0.32512009143829346, 0.8727200627326965]
'''