'''
network.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

# Tensorflow dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

eps = 1e-7

# input width
W = 224

def conv_block(x, filters, kernel_size=3, strides=1, padding='same'):
  x = tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=strides,
    padding=padding,
    use_bias=False)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation("relu")(x)
  return x

def create_model(input_shape=(W,W,3)):
  base = tf.keras.applications.MobileNetV2(input_shape, include_top=False, weights='imagenet')
  # for layer in base.layers[:54]:
  #   layer.trainable = False

  x = base.layers[80].output # 14x14x64
  # regression
  s0 = conv_block(x , 16, 3, 1) # 14x14x16
  s1 = conv_block(s0, 32, 3, 2) # 7x7x32
  # s2 = tf.keras.layers.Conv2D(136, 7, strides=1, padding='valid')(s1)
  # outputs = tf.keras.layers.Flatten()(s2)
  y = tf.keras.layers.Flatten()(s1)
  outputs = tf.keras.layers.Dense(136)(y)

  model = tf.keras.models.Model(inputs=base.input, outputs=outputs)
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
  model.compile(optimizer=opt, loss="mse", metrics=["acc"])

  return model