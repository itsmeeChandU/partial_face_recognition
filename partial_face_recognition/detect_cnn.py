import cv2
from PIL import Image

import face_recognition


def detect_face_using_cnn(image):
    # Load the image into memory
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Applying Neural Networks for Face Detection.......")
    face_locations = face_recognition.face_locations(image, model="cnn")
    detected_faces = []
    detected_face_ids = []
    print("Total faces {} .".format(len(face_locations)))
    i = 1
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("Location of face: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))
        # writing images in detected folder

        face_image = image[top - 50:bottom + 50, left - 10:right + 30]

        pil_image = Image.fromarray(face_image)
        detected_faces.append(pil_image)
        detected_face_ids.append(str(i)+'.jpg')
        # pil_image = pil_image.resize((240, 240))
        pil_image.save('Detected_faces/' + str(i) + '.jpg', 'JPEG', quality=100)
        # pil_image.show()
        i += 1
    return face_locations, detected_faces
