import cv2
import dlib
import numpy as np
import func_tools
import imutils
from imutils import face_utils
def MAR(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def mouth(cap):
    MAR_THRESH = 0.5
    COUNTER_MOUTH = 0
    TOTAL_MOUTH = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
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

            frame = func_tools.draw_face_contour(frame, shape)

            Mouth = shape[mStart:mEnd]
            mar = MAR(Mouth)

            if mar > MAR_THRESH:
                COUNTER_MOUTH += 1
            else:
                if COUNTER_MOUTH >= 2:
                    TOTAL_MOUTH += 1
                COUNTER_MOUTH = 0


        if TOTAL_MOUTH == 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    return True
if __name__ == '__main__':
    mouth()
