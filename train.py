'''
train.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python train.py
'''
from __future__ import print_function

import os
import numpy as np
import cv2
import pickle
import gzip
import random
import matplotlib.pyplot as plt

import tensorflow as tf

from network import create_model
from datagenerator import DataGenerator

# Tensorflow dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

def load_data(bin_path):
  if os.path.isfile(bin_path):
    with gzip.open(bin_path, 'rb') as f:
      data = pickle.load(f)
      x_data, y_data = data["x_data"], data["y_data"]
      # shuffle
      idx = np.arange(len(x_data)); np.random.shuffle(idx)
      x_train, y_train = x_data[idx], y_data[idx]
      return x_train, y_train
  exit("Could not load data at: "+bin_path)

if __name__ == '__main__':
  batch_size = 16
  validation_split = .3
  use_generator = True

  check_path = 'model/weights.h5'
  save_path = 'model/network.h5'

  ###################
  # Create model
  print('-'*30)
  print('Create model')
  model = create_model()
  model.summary()
  exit("quit")

  #########################
  # Load data for training
  print('-'*30)
  print('Load train data')
  x_data, y_data = load_data('./data/ibug300_raw.pickle')

  # original landmark position is 0 ~ 1
  # shift all points so that 0.5,0.5 is the origin
  if not use_generator:
    y_data -= .5

  # size of total dataset
  N = len(y_data)
  Nt = int(N * (1. - validation_split)) # training data split
  print(x_data.shape, y_data.shape)

  ###################
  # Start Training
  print('-'*30)
  print('Training start')
  # callbacks
  checkpoint = tf.keras.callbacks.ModelCheckpoint(
    check_path, monitor='val_loss', save_best_only=True)
  earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, verbose=0, mode='auto')

  # Initially, model.fit was conducted within the loop,
  # but it was changed to manually adjust schedule index
  # due to a problem of keras's memory leak.
  # (i.e. memory not enough error raised when model.fit called 2nd times)
  # for lr, iterations in schedule:
  #   model.fit(...)

  # loading weights of the last check point
  if os.path.isfile(check_path):
    model.load_weights(check_path)

  # training schedule tuple of learning rate and iterations
  schedule = [(1e-3, 50000), (2e-4, 40000), (1e-4, 30000)]
  schedule_idx = 0 # <-- modify here manually

  if use_generator:
    # data generators
    training_generator = DataGenerator(x_data[:Nt], y_data[:Nt], batch_size=batch_size)
    validation_generator = DataGenerator(x_data[Nt:], y_data[Nt:], batch_size=batch_size)
    
    lr, iterations = schedule[schedule_idx]
    epochs = int(iterations * batch_size / Nt)
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
    hist = model.fit(
      training_generator,
      None,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      use_multiprocessing=True,
      workers=2,
      callbacks=[checkpoint, earlystop]
    )
  else:
    lr, iterations = schedule[schedule_idx]
    epochs = int(iterations * batch_size / Nt)
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)
    hist = model.fit(
      x_data,
      y_data,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=validation_split,
      callbacks=[checkpoint, earlystop]
    )
  
  if not os.path.exists('model'):
    os.makedirs('model')
  model.save(save_path)
  print('Trainig finished.')

  # Loss History
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('rate')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()