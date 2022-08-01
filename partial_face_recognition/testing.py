import pickle
import sys
import cv2

from partial_face_recognition.utils.helpers import *
from partial_face_recognition.utils.constants import *

path_train = TRAIN_IMAGES_PATH  # to show the results of the algorithm (detecetion and matching)
path_test = TEST_IMAGES_PATH


def calculate_results(test_features, train_features):
    """
    This function is used to compare the features of the detected images with the features of the training images.
    """

    test_images_status = []  # Status  0: same person, 1: different person, 2 represents of the unknown face
    best_match = []  # add the name of the best match to the list of names
    not_detected_names = []  # add the name of the image to the list of images that are not detected
    for filename_test in test_features.keys():
        print("Applying our algorithm for " + filename_test)
        current_image = show_image(filename_test, path_test)
        cv2.imshow("Image", current_image)
        cv2.waitKey(0)
        print("Showing the extracted features of images")
        extract_features(filename_test, path_test)
        print("Matching the Features with our data base")
        c = 0
        for des_test in test_features[filename_test]:
            print("Comparing image" + filename_test + " " + str(c))
            c += 1
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)  # bug : need to be created after each loop
            result = {}
            for filename_train in train_features.keys():
                for des_train in train_features[filename_train]:  # for each image in the training set
                    good = []
                    ctr = 0
                    matches = flann.knnMatch(des_train, des_test, k=2)  # find the 2 nearest neighbors
                    try:
                        for m, n in matches:
                            if m.distance < 0.8 * n.distance:
                                good.append([m])
                                ctr += 1
                    except:
                        pass

                    if filename_train in result.keys():
                        if result[
                            filename_train] < ctr:  # if the current image is more similar to the train image than the previous one
                            result[filename_train] = ctr
                    else:
                        result[filename_train] = ctr

            # print(result.items())
            best_match = max(result, key=lambda x: result[x])  # get the best match

            if (result[best_match] < 80):  # if the best match is less than 80%
                print("Not detected")
                test_images_status.append(2)  # 2 represents of the unknown face
                not_detected_names.append(
                    filename_test)  # add the name of the image to the list of images that are not detected
                continue
            else:
                print("Detected person " + best_match)
                show_image(best_match, path_train)
                print("Showing the Matched Features in both images")
                try:
                    draw_detected_face(show_image(filename_test, path_test),
                                       show_image(best_match, path_train))
                except:
                    draw_detected_face(show_image(filename_test, path_test),
                                       show_image(best_match, path_train))

            if best_match == filename_test:
                test_images_status.append(0)  # 0 represents of the same person
            else:
                test_images_status.append(1)  # 1 represents of different person
            best_match.append(best_match)  # add the name of the best match to the list of names
            not_detected_names.append(filename_test)

    return test_images_status, best_match, not_detected_names


def calculate_accuracy(test_images_status):
    correct = 0
    for i in range(len(test_images_status)):
        if test_images_status[i] == 0:
            correct += 1
    print("Accuracy: " + str(correct / len(test_images_status)))
    return correct / len(test_images_status)


def save_results(test_images_status, best_match, not_detected_names):  # save the results of the algorithm
    filename = "y_res.txt"
    print("y_res....../")
    file = open(filename, 'w')
    for i in test_images_status:
        file.write("%s\n" % i)
    file.close()
    filename = "y_name.txt"
    print("y_name....../")
    file = open(filename, 'w')
    for i in best_match:
        file.write("%s\n" % i)
    file.close()
    filename = "y_req.txt"
    print("y_res....../")
    file = open(filename, 'w')
    for i in not_detected_names:
        file.write("%s\n" % i)
    file.close()
