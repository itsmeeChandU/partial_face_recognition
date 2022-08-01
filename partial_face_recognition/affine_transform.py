import cv2
import dlib
from imutils.face_utils import FaceAligner
from skimage import io


def face_align(img): # align the face
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        faceAligned = fa.align(img, gray, rect)
        return faceAligned
