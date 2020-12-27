'''
prepare.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python prepare.py <dataset dir>
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import pickle
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from xml.etree.ElementTree import parse
from utils import draw

W = 224

def load_ibug300W(path):
  ''' Parse iBug300W large dataset
  Args:
    path (str) : A path where the iBug300W dataset decompressed
  
  Returns:
    list : A list of image path (str), bounding boxes (2-D list) and face landmarks (3-D list)
           Bounding box can be parsed as left, top, right, bottom (in pixels)
  '''
  print("loading iBug300W large data set...")

  # check if path is available
  if not os.path.isdir(path):
    exit("dataset is not exists: " + path)
  
  xml_file = os.path.join(path, "labels_ibug_300W.xml")
  if not os.path.isfile(xml_file):
    exit('Could not open metadata file: '+xml_file)
  
  xml = parse(xml_file)
  root = xml.getroot()
  images = root.find("images")
  dic = {}
  for i,image in enumerate( images.findall("image") ):
    img_path = os.path.join(path, image.attrib["file"])
    if img_path not in dic:
      dic[img_path] = [[], []]
    for box in image.findall("box"):
      bb = [
        int(box.attrib["left"]),
        int(box.attrib["top"]),
        int(box.attrib["width"]),
        int(box.attrib["height"])
      ]
      if bb[2] < 1 or bb[3] < 1:
        print("skip: ", box)
        continue
      pts = np.zeros((68,2), dtype=np.float32)
      for part in box.findall("part"):
        p = int(part.attrib["name"])
        pts[p,0] = float(part.attrib["x"])
        pts[p,1] = float(part.attrib["y"])
      dic[img_path][0].append(bb)
      dic[img_path][1].append(pts)
  
  data = []
  for img_path in dic:
    bb, pts = dic[img_path]
    data.append([img_path, bb, pts])
  return data

if __name__ == '__main__':
  bin_path = './data/ibug300_raw.pickle'

  # Check input parameters
  if len(sys.argv) != 2:
    exit('usage: python {} <dataset dir>'.format(sys.argv[0]))
  
  src_path = sys.argv[1]

  print('-'*30)
  print('load meta data')
  data = load_ibug300W(src_path)

  if not data:
    exit('no data for pre-processing')

  print('-'*30)
  print('load image and annotation')
  x_data, y_data = [], []
  for i in tqdm(range(len(data))):
    img_path, boxes, landmarks = data[i]
    # image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      print("could not open file: " + img_path); continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]

    for box, pts in zip(boxes, landmarks):
      l,t,w,h = box
      pts = (pts - [l,t]) / [w,h]
      # zero padding
      y0,y1,x0,x1 = max(0,t), min(height,t+h), max(0,l), min(width,l+w)
      if l<0 or t<0 or l+w > width or t+h > height:
        crop = np.zeros((h,w,3), dtype=np.uint8)
        ox,oy = x0-l, y0-t
        crop[oy:oy+y1-y0, ox:ox+x1-x0] = img[y0:y1, x0:x1]
      else:
        crop = img[y0:y1, x0:x1]

      x = cv2.resize(crop, (W,W))
      y = pts.ravel()
      # if i < 100:
      #   pass
      # elif i < 130:
      #   draw(x, pts)
      # else:
      #   exit("quit")
      
      x_data.append(x)
      y_data.append(y)
  
  x_data = np.array(x_data)
  y_data = np.array(y_data, dtype=np.float32)
  print(x_data.shape, y_data.shape)

  print('-'*30)
  print('save to file')
  # save raw blob
  with gzip.open(bin_path, 'wb') as f:
    pickle.dump({ "x_data":x_data, "y_data":y_data }, f)
  
  print('done')