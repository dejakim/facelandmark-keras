'''
utils.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_point(pt, sx=1., sy=1.):
  return (int(pt[0] * sx), int(pt[1] * sy))

def draw(img, pts):
  h,w = img.shape[:2]
  for pt in pts:
    img = cv2.circle(img, to_point(pt,w,h), 2, (0,255,0), thickness=-1)
  plt.imshow(img)
  plt.show()
