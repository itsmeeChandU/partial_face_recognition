import os

import cv2
import numpy as np


def load_images_from_folder(folder):
    c = 0
    images, names = [], []
    print("Reading the Images from directory........")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
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
