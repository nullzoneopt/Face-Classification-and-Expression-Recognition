import cv2
import sys
import numpy
import os
from PIL import Image, ImageOps
import time

size = 4

front_haar = "..\\haar_classifiers\\haarcascade_frontalface_default.xml"  # "haarcascade_profileface.xml"# 'haarcascade_frontalface_default.xml'
side_haar = "..\\haar_classifiers\\haarcascade_profileface.xml"  # "haarcascade_profileface.xml"# 'haarcascade_frontalface_default.xml'

fn_dir = 'database'  # All the faces data will be present this folder
fn_name = 'vidit_test'# sys.argv[1]

path = os.path.join(fn_dir, fn_name)
if not os.path.isdir(path):
    os.mkdir(path)
(im_width, im_height) = (112, 92)
front_cascade = cv2.CascadeClassifier(front_haar)
side_cascade = cv2.CascadeClassifier(side_haar)

webcam = cv2.VideoCapture(0)  # '0' is use for my webcam, if you've any other
# The program loops until it has 30 images of the face.
count = 0
while count < 20:
    # time.sleep(0.100)
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    if len(front_cascade.detectMultiScale(mini)) == 1 and len(side_cascade.detectMultiScale(mini)) == 0:

        faces = front_cascade.detectMultiScale(mini)
        # print("front")
    elif len(front_cascade.detectMultiScale(mini)) == 0 and len(side_cascade.detectMultiScale(mini)) == 1:
        faces = side_cascade.detectMultiScale(mini)
        # print("side")
    else:
        faces = front_cascade.detectMultiScale(mini)
        # print(len(front_cascade.detectMultiScale(mini)) ,len(side_cascade.detectMultiScale(mini)))
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
        face_i = faces[0]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                      if n[0] != '.'] + [0])[-1] + 1
        cv2.imwrite('%s/%s.jpg' % (path, pin), face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0))
        imm = Image.open('%s/%s.jpg' % (path, pin))
        # print(imm)
        pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                      if n[0] != '.'] + [0])[-1] + 2
        im_mirror = ImageOps.mirror(imm)
        im_mirror.save('%s/%s.jpg' % (path, pin), quality=95)
        count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(5)
    if key == 27:
        break