"""Main module."""

from utils.helpers import *
from utils.constants import *
from .testing import *
from .calculate_features import *


if __name__ == "__main__":

    X_train, y_train = load_images_from_folder(TRAIN_IMAGES_PATH)
    print("Data Base Description of training")
    print(len(X_train), len(y_train))
    print("Calculating the features of the training set........./")
    calculate_features(X_train, y_train, filename="train_features")

    X_test, y_test = load_images_from_folder(TEST_IMAGES_PATH)
    print("Data Base Description of testing")
    print(len(X_test), len(y_test))
    print("Calculating the features of testing images....../")
    calculate_features(X_test, y_test, filename="test_features")

    pickle_in = open("test_features.pickle", "rb")
    test_ft = pickle.load(pickle_in)

    pickle_in = open("train_features.pickle", "rb")
    train_ft = pickle.load(pickle_in)

    print("Calculating the results of the testing set........./")
    test_images_status, best_match, not_detected_images = calculate_results(test_ft, train_ft)
    print("Results of the testing set")
    print(test_images_status)
    print("Best match")
    print(best_match)
    print("Not detected images")
    print(not_detected_images)

    print("Showing the images of the testing set........../")
    for i in range(len(test_images_status)):
        if test_images_status[i] == "Not Detected":
            img = show_image(i, path=TEST_IMAGES_PATH)
            cv2.imshow("Not Detected", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            img = show_image(i, path=TEST_IMAGES_PATH)
            cv2.imshow("Detected", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("Accuracy of the system: " + str(calculate_accuracy(test_images_status)))

    print("Saving the results of the testing set........./")
    save_results(test_images_status, best_match, not_detected_images)



