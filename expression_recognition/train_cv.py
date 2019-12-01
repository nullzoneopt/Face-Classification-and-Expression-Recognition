import cv2
import numpy
from utils import helper as helper
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def evaluate(model, X_train, y_train):
    cnt = 0
    correct = 0
    incorrect = 0
    for image in X_train:
        pred = model.predict(image)
        if pred == y_train[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    return correct / cnt


haar_file = '..\\haar_classifiers\\haarcascade_frontalface_default.xml'
haar_file_side = '..\\haar_classifiers\\haarcascade_profileface.xml'
fn_dir = 'database'

# Create a list of images and a list of corresponding names
images, labels = helper.generate_dataset(fn_dir)

# Create array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 5 fold CV
cv = KFold(n_splits=5, random_state=42, shuffle=True)
fold = 1
training_fold_accuracy = {}
testing_fold_accuracy = {}
testing_accuracy = {}
model = cv2.face.LBPHFaceRecognizer_create()

for train_index, test_index in cv.split(x_train):  # OpenCV trains a model from the images
    X_train_fold, X_test_fold, y_train_fold, y_test_fold = x_train[train_index], x_train[test_index], \
                                                           y_train[train_index], y_train[test_index]
    model.train(X_train_fold, y_train_fold)
    training_fold_accuracy[fold] = evaluate(model, X_train_fold, y_train_fold)
    testing_fold_accuracy[fold] = evaluate(model, X_test_fold, y_test_fold)
    testing_accuracy[fold] = evaluate(model, x_test, y_test)
    model.save('..\\trained_models\\expression_classification' + str(fold) + '.xml')
    print("saved model: %s" % (str(fold)))
    fold += 1

print('Testing Accuracy')
print(testing_accuracy)
print('Training Fold')
print(training_fold_accuracy)
print('Testing Fold')
print(testing_fold_accuracy)
