'''
app.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np
import cv2

from utils import to_point
from detector import Detector

xml_path = './data/haarcascade_frontalface_default.xml'

def crop(img):
  h,w = img.shape[:2]
  ow = min(h,w)
  ox,oy = int((w-ow)/2), int((h-ow)/2)
  return img[oy:oy+ow, ox:ox+ow]

if __name__ == '__main__':
  # Download trained xml file for OpenCV2 face detection
  if not os.path.isfile(xml_path):
    req = requests.get('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
    with open(xml_path, 'wb') as f:
      f.write(req.content)
  
  if not os.path.isfile(xml_path):
    exit('could not download trained xml for face detecion')
  
  # Load model
  face_detector = cv2.CascadeClassifier(xml_path)
  detector = Detector()

  # camera open
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  # main loop
  while True:
    ret, frame = capture.read()

    frame = crop(frame)
    # Convert color space
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.pyrDown(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Inference
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    for x,y,w,h in faces:
      x,y,w,h = x*2, y*2, w*2, h*2
      frame = cv2.rectangle(frame, to_point([x,y]), to_point([x+w,y+h]), (255,0,0), 2)

      pts = detector.inference(img[y:y+h, x:x+w])
      pts *= [w, h]
      pts += [x+w/2, y+h/2]
      for pt in pts:
        frame = cv2.circle(frame, to_point(pt), 3, (0,255,0), thickness=-1)
    
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0: break

  capture.release()
  cv2.destroyAllWindows()
