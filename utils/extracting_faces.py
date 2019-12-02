import cv2
import glob

face_haar_one = cv2.CascadeClassifier("..\\haar_classifiers\\haarcascade_frontalface_default.xml")
face_haar_two = cv2.CascadeClassifier("..\\haar_classifiers\\haarcascade_frontalface_alt2.xml")
face_haar_three = cv2.CascadeClassifier("..\\haar_classifiers\\haarcascade_frontalface_alt.xml")
face_haar_four = cv2.CascadeClassifier("..\\haar_classifiers\\haarcascade_frontalface_alt_tree.xml")

parent_source_directory = 'source'
parent_destination_directory = 'destination'

(image_width, image_height) = (112, 92)


def find_face(gray):
    face = face_haar_one.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        return face
    face = face_haar_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        return face
    face = face_haar_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        return face
    face = face_haar_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        return face


def write_valid_images(emotion, parent_source_directory, parent_destination_directory):
    files = glob.glob("%s\\%s\\*" % (parent_source_directory, emotion))
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        # Detect face using 4 different classifiers
        face = find_face(gray)
        # Cut and save face
        for (x, y, w, h) in face:
            print("face found in file: %s" % f)
            gray = gray[y:y + h, x:x + w]
            try:
                out = cv2.resize(gray, (image_width, image_height))
                cv2.imwrite("%s\\%s\\%s.jpg" % (parent_destination_directory, emotion, filenumber), out)
            except:
                pass
        filenumber += 1


sub_folder_seggregation = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
for emotion in sub_folder_seggregation:
    write_valid_images(emotion, parent_source_directory, parent_destination_directory)  # Call function
print('completed')
