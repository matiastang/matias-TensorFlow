'''
Author: matiastang
Date: 2022-08-01 11:20:57
LastEditors: matiastang
LastEditTime: 2022-08-01 17:23:53
FilePath: /matias-TensorFlow/src/python/tensorflow_install_test.py
Description: 测试tensorflow安装
'''
#!/usr/bin/python3
#coding=utf-8

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())

'''
`conda`环境运行结果
$ python3 /Users/matias/matias/MT/MTGithub/matias-TensorFlow/src/python/tensorflow_install_test.py
2.9.2
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
'''