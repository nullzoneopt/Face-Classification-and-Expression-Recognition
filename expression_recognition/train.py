import cv2
import sys
import numpy
import os
from PIL import Image, ImageOps

haar_file = '..\\haar_classifiers\\haarcascade_frontalface_default.xml'
haar_file_side = '..\\haar_classifiers\\haarcascade_profileface.xml'
datasets = 'database'
fn_dir = 'database'

# Part 1: Training model
print('Training starts please wait....')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(im_width, im_height) = (112, 92)
# (im_width, im_height) = (48, 48)

# Create array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.save('..\\trained_models\\expression_classification.xml')
print("saved")