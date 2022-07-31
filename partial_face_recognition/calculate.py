import pickle

import cv2

import orb_output as op
import read_images

print("Importing the data set...")
path_test = "D:/test"
path_train = "D:/battleground/Main Weapon/Codes/Final_demo/Data_set/ar_faces latest"

X_train, y_train = read_images.load_images_from_folder(path_train)
X_test, y_test = read_images.load_images_from_folder(path_test)

# X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Data Base Description of training and testing")
print(len(X_train), len(X_test), len(y_train), len(y_test))

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

print("Calculating the features of Detected images....../")
for (image, name) in zip(X_test, y_test):
    kp1 = orb.detect(image, None)
    kp1, des1 = orb.compute(image, kp1)
    if name in feature_test:
        feature_test[name].append(des1)
    else:
        feature_test[name] = []
        feature_test[name].append(des1)

filename = "test_features.pickle"
print("storing the features of Detected images....../")
file = open(filename, 'wb')
pickle.dump(feature_test, file)
file.close()

orb = cv2.ORB_create()

print("Calculating the results of the training set........./")

pickle_in = open("test_features.pickle", "rb")
test_ft = pickle.load(pickle_in)

pickle_in = open("train_features.pickle", "rb")
train_ft = pickle.load(pickle_in)
y_res = []
y_name = []
y_req = []

print(test_ft.keys())
print(train_ft.keys())


def match_results(test_features=test_ft, train_features=train_ft):
    for filename_test in test_features.keys():
        print("Applying our algorithm for " + filename_test)
        Im = op.show_image(filename_test, path_test)
        # cv2.imshow("Image",Im)
        cv2.waitKey(0)
        print("Showing the extracted features of images")
        op.extract_features(filename_test, path_test)
        print("Matching the Features with our data base")
        c = 0
        for des_test in test_features[filename_test]:
            print("testing image" + filename_test + " " + str(c))
            c += 1
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            result = {}
            for filename_train in train_features.keys():
                for des_train in train_features[filename_train]:
                    good = []
                    ctr = 0
                    matches = flann.knnMatch(des_train, des_test, k=2)
                    try:
                        for m, n in matches:
                            if (m.distance < 0.8 * n.distance):  # Apply the ratio test as per the paper
                                good.append([m])
                                ctr += 1
                    except:
                        pass

                    if filename_train in result.keys():
                        if result[filename_train] < ctr:
                            result[filename_train] = ctr
                    else:
                        result[filename_train] = ctr

            # print(result.items())
            best_match = max(result, key=lambda x: result[x])

            if (result[best_match] < 80):
                print("Not detected")
                y_res.append(2)
                y_req.append(filename_test)
                continue
            else:
                print("Detected person " + best_match)
                Im = op.show_image(best_match, path_train)
                cv2.imshow(Im)
                cv2.waitKey(0)
                print("Showing the Matched Features in both images")
                op.draw_detected_face(op.show_image(filename_test, path_test), op.show_image(best_match, path_train))

            if best_match == filename_test:
                y_res.append(0)
            else:
                y_res.append(1)
            y_name.append(best_match)
            y_req.append(filename_test)

    print(y_res)
    print(y_name)
    print(y_req)
    return y_res, y_name, y_req


y_res, y_name, y_req = match_results()

filename = "y_res.txt"
print("y_res....../")
file = open(filename, 'w')
for i in y_res:
    file.write("%s\n" % i)
file.close()
filename = "y_name.txt"
print("y_name....../")
file = open(filename, 'w')
for i in y_name:
    file.write("%s\n" % i)
file.close()
filename = "y_req.txt"
print("y_res....../")
file = open(filename, 'w')
for i in y_req:
    file.write("%s\n" % i)
file.close()
