import cv2
import numpy
from utils import helper as helper

haar_file = '..\\haar_classifiers\\haarcascade_frontalface_default.xml'
haar_file_side = '..\\haar_classifiers\\haarcascade_profileface.xml'
fn_dir = 'database'

# Create a list of images and a list of corresponding names
images, labels = helper.generate_dataset(fn_dir)

# Create array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
model.save('..\\trained_models\\face_recognition.xml')
print("saved")
