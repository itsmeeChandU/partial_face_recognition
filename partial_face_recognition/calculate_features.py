from utils.helpers import *
from .detect_cnn import *


def calculate_features(X_train, y_train, filename):
    feature_train = {}
    orb = cv2.ORB_create()
    print("Calculating features of training set..../")

    for (image, name) in zip(X_train, y_train):
        _, detected_faces = detect_face_using_cnn(image)
        if len(detected_faces) == 0:
            print("No face detected in " + name)
            continue
        else:
            image = detected_faces[0]
            kp1 = orb.detect(image, None)
            kp1, des1 = orb.compute(image, kp1)
            if name in feature_train:
                feature_train[name].append(des1)
            else:
                feature_train[name] = []
                feature_train[name].append(des1)

    filename = str(filename) + ".pickle"
    print("storing the features of training set....../")
    file = open(filename, 'wb')
    pickle.dump(feature_train, file) # dumping the data into a pickle file
    file.close()
    return feature_train, filename
