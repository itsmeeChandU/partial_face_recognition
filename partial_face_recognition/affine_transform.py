import cv2
import dlib
from imutils.face_utils import FaceAligner
from skimage import io

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "D:/battleground/Main Weapon/Codes/Collage_demo/face_recognition/face_recognition_models/shape_predictor_68_face_landmarks.dat"
predictor_model2 = "D:/battleground/Main Weapon/Codes/Collage_demo/face_recognition/face_recognition_models/shape_predictor_5_face_landmarks.dat"

# Take the image file name from the command line
# file_name = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

#face_aligner = openface.AlignDlib(predictor_model)

face_aligner = FaceAligner(face_pose_predictor, desiredFaceWidth=256)

win = dlib.image_window()

# Take the image file name from the command line
file_name = 'D:/m-002-4.jpg'

# Load the image
image = io.imread(file_name)
image1 = cv2.imread(file_name)

# Run the HOG face detector on the image data
detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# Show the desktop window with the image
win.set_image(image)


# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):
    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                             face_rect.right(), face_rect.bottom()))
    # Draw a box around each face we found
    win.add_overlay(face_rect)

    # Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)

    print(face_rect)
    # Draw the face landmarks on the screen.
    win.add_overlay(pose_landmarks)
    alignedFace = face_aligner.align(534, image, face_rect)
    alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
    # alignedFace=alignedFace+50
    cv2.imwrite("aligned_face_{}.jpg".format(1), alignedFace)

dlib.hit_enter_to_continue()