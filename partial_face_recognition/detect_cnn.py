import cv2
from PIL import Image

import face_recognition


def apply_cnn(image_path, upsample_times=0):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Applying Neural Networks for Face Detection.......")
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample_times, model="cnn")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    i = 1
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # writing images in detected folder

        face_image = image[top - 50:bottom + 50, left - 10:right + 30]

        pil_image = Image.fromarray(face_image)
        # pil_image = pil_image.resize((240, 240))
        pil_image.save('Detected_faces/' + str(i) + '.jpg', 'JPEG', quality=100)
        # pil_image.show()
        i += 1
    return face_locations

# apply_cnn("D:/battleground/Main Weapon/Codes/Final_demo/test image.jpg")
