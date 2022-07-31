import pickle
import sys

import cv2

import read_images

path_train = sys.argv[1]

print("Importing the data set...")

X_train, y_train = read_images.load_images_from_folder(path_train)

print("Data Base Description of training")
print(len(X_train), len(y_train))

feature_train = {}
feature_test = {}
orb = cv2.ORB_create()

print("Calculating features of training set..../")

for (image, name) in zip(X_train, y_train):

    kp1 = orb.detect(image, None)
    kp1, des1 = orb.compute(image, kp1)
    if name in feature_train:
        feature_train[name].append(des1)
    else:
        feature_train[name] = []
        feature_train[name].append(des1)

filename = "train_features.pickle"
print("storing the features of training set....../")
file = open(filename, 'wb')
pickle.dump(feature_train, file)
file.close()
