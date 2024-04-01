import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils
def EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def eye(cap):
    EAR_THRESH = 0.3
    EYE_close = 2
    count_eye = 0
    total = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = EAR(leftEye) + EAR(rightEye) / 2.0

            if ear < EAR_THRESH:
                count_eye += 1
            else:
                if count_eye >= EYE_close:
                    total += 1
                count_eye = 0

        if total == 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    eye()
