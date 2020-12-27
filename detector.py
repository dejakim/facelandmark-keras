'''
detector.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python detector.py <input file>
'''

from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import requests
from matplotlib import pyplot as plt

import tensorflow as tf

from utils import draw, to_point


class Detector:
  def __init__(self):
    # load saved model
    model = tf.keras.models.load_model('./model/network.h5')

    W,H,C = model.input.get_shape()[1:4] # None, width, height, channel
    print('model loaded: input shape=', (W,H,C))
    self._model = model
    self._W = W
    self._C = C
    
  def inference(self, img):
    W,C = self._W, self._C
    # Resize image
    h,w = img.shape[:2]
    src = cv2.resize(img, (W,W)).reshape(-1,W,W,C)
    # Inference
    infer = self._model.predict(src)[0]
    landmarks = infer.reshape(-1,2)
    return landmarks


if __name__ == '__main__':
  xml_path = './data/haarcascade_frontalface_default.xml'

  # Check input parameters
  if len(sys.argv) != 2:
    exit('usage: python {} <input file>'.format(sys.argv[0]))
  
  if not os.path.isfile(xml_path):
    req = requests.get('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
    with open(xml_path, 'wb') as f:
      f.write(req.content)
  
  if not os.path.isfile(xml_path):
    exit('could not download trained xml for face detecion')
  
  src_path = sys.argv[1]

  # Load source image
  img = cv2.imread(src_path, cv2.IMREAD_COLOR)
  if img is None:
    exit('could not open file: {}'.format(src_path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  height, width = img.shape[:2]

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_detector = cv2.CascadeClassifier(xml_path)
  faces = face_detector.detectMultiScale(gray, 1.1, 4)
  if len(faces) < 1:
    exit('could not find face!')

  # Load model
  detector = Detector()

  # Inference
  radius = max(2, min(width,height)//150)
  for face in faces:
    x,y,w,h = face
    # avoid too small face
    if max(w,h)/min(width,height) < .05:
      continue
    # draw rect
    img = cv2.rectangle(img, to_point([x,y]), to_point([x+w,y+h]), (255,0,0), 2)
    # predict landmarks
    pts = detector.inference(img[y:y+h, x:x+w])
    pts *= [w, h]
    pts += [x+w/2, y+h/2]
    # draw landmarks
    for pt in pts:
      img = cv2.circle(img, to_point(pt), radius, (0,255,0), thickness=-1)
  
  plt.imshow(img)
  plt.show()
