import cv2
from utils import helper as helper
# import sys
# import numpy
# import os
# from PIL import Image, ImageOps

haar_file = '..\\haar_classifiers\\haarcascade_frontalface_default.xml'
haar_file_side = '..\\haar_classifiers\\haarcascade_profileface.xml'
fn_dir = 'database'
model = cv2.face.LBPHFaceRecognizer_create()
model.read('..\\trained_models\\expression_classification.xml')

# Create a list of images and a list of corresponding names
names = helper.generate_classification_labels(fn_dir)
(im_width, im_height) = (112, 92)

def identify(faces, model):
    # print("here")
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_width))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print('pred', prediction)
        if prediction[1] < 100:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]],
                                           prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))



face_cascade = cv2.CascadeClassifier(haar_file)
side_face_cascade = cv2.CascadeClassifier(haar_file_side)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    side_faces = side_face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1 and len(side_faces) == 0:
        identify(faces, model)
        pass
    elif len(faces) == 0 and len(side_faces) == 1:
        identify(side_faces, model)
        pass
    elif len(faces) == 1 and len(side_faces) == 1:
        identify(faces, model)
        pass
    else:
        immm = cv2.flip(im, 1)
        gray = cv2.cvtColor(immm, cv2.COLOR_BGR2GRAY)
        # print(gray)
        side_faces = side_face_cascade.detectMultiScale(gray, 1.3, 5)
        identify(side_faces, model)

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break