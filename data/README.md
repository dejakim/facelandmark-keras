# Dataset preparation
To train this model, I used official [DLib dataset](http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz) which can be downloaded from following commands.

```
$ cd data
$ wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
$ tar -xvzf ibug_300W_large_face_landmark_dataset.tar.gz
```
The dataset contains annotated labels in xml files which provides 68 landmarks for each face bounding box.  
To save training time in CoLab environment, all images and labels are zipped in one blob file in this directory. Thus, you need to run [prepare.py](../prepare.py) in advance.

```
$ python prepare.py ./data/ibug_300W_large_face_landmark_dataset
```

[prepare.py](../prepare.py) crops and resizes only the face area from the image, and returns the landmarks of each face area in pairs.

| x_data | y_data |
|--------|--------|
| face 1 (224x224x3) | landmarks (x0,y0,x2,y2,...x85,y85) |
| face 2 (224x224x3) | landmarks (x0,y0,x2,y2,...x85,y85) |
| ... | ... |
