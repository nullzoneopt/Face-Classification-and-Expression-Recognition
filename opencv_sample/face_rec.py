# facerec.py
import cv2, sys, numpy, os, pyttsx3, time
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'


# Part 1: Create fisherRecognizer
print('Training...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

# OpenCV trains a model from the images

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Part 2: use LBPHFace recognizer on camera frame
face_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
(im_width, im_height) = (112, 92)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print ('pred', prediction)
        if prediction[1] <100:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]],
                                           prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'not recognized', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break