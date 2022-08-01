import os
import cv2
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt
from .constants import TRAIN_IMAGES_PATH, TEST_IMAGES_PATH
import pickle

path_test = TEST_IMAGES_PATH
path_train = TRAIN_IMAGES_PATH


def load_images_from_folder(folder):
    c = 0
    images, names = [], []
    print("Reading the Images from directory........")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        # img = io.imread(os.path.join(folder, filename), as_gray=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(np.asarray(img, dtype=np.uint8))
            words = filename
            try:
                split_words = words.split("-")
                print(split_words[0] + split_words[1])
                names.append(split_words[0] + split_words[1])
            except:
                c = c + 1
                names.append(str(c))

    return images, names


def show_image(img_name, path=path_test):
    for filename in os.listdir(path):
        c = 0
        words = filename
        # print(filename)
        try:
            l = words.split("-")
            l = l[0] + l[1]
        except:
            c = c + 1
        if l == img_name or c == img_name:
            img1 = cv2.imread(os.path.join(path, filename))
            cv2.imshow("pic", img1)
            # dlib.hit_enter_to_continue()
            return img1


def extract_features(img_name, path=path_train):
    for filename in os.listdir(path):
        c = 0
        words = filename
        try:
            l = words.split("-")
            l = l[0] + l[1]
        except:
            c = c + 1
        if l == img_name or c == img_name:
            orb = cv2.ORB_create()
            img1 = cv2.imread(os.path.join(path, filename))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp1 = orb.detect(img1, None)
            # compute the descriptors with ORB
            kp1, des1 = orb.compute(img1, kp1)

            # draw only keypoints location,not size and orientation
            op1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
            plt.imshow(op1)
            plt.show()


def draw_detected_face(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create() # create ORB detector

    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params) # bug : need to instantiate with random seed
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))] # create a mask with zeros
    for i, (m, n) in enumerate(matches): # m is the best match, n is the second best match
        if m.distance < 0.9 * n.distance: # if the distance is less than 10% of the second best, then it is a good match
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0), # draw matches in green
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params) # draw the matches
    plt.imshow(img3, ), plt.show() # show the image


