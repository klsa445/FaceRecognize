# coding:utf-8
import cv2
import dlib
import imutils
import numpy as np
import func_tools

def nose_jaw_distance(nose, jaw):
    # 计算鼻子到左右脸边界的欧式距离
    face_left1 = np.linalg.norm(nose[0] - jaw[0])  # 27, 0
    face_right1 = np.linalg.norm(nose[0] - jaw[16])  # 27, 16
    face_left2 = np.linalg.norm(nose[3] - jaw[2])  # 30, 2
    face_right2 = np.linalg.norm(nose[3] - jaw[14])  # 30, 14

    face_distance = (face_left1, face_right1, face_left2, face_right2)

    return face_distance

def main():
    distance_left = 0
    distance_right = 0
    TOTAL_FACE = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']

    cap = cv2.VideoCapture(0)

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

            nose = shape[nStart:nEnd]
            jaw = shape[jStart:jEnd]
            NOSE_JAW_Distance = nose_jaw_distance(nose, jaw)
            face_left1 = NOSE_JAW_Distance[0]
            face_right1 = NOSE_JAW_Distance[1]
            face_left2 = NOSE_JAW_Distance[2]
            face_right2 = NOSE_JAW_Distance[3]

            if face_left1 >= face_right1 + 20 and face_left2 >= face_right2 + 20:
                distance_left += 1
            if face_right1 >= face_left1 + 20 and face_right2 >= face_left2 + 20:
                distance_right += 1
            if distance_left != 0 and distance_right != 0:
                TOTAL_FACE += 1
                distance_right = 0
                distance_left = 0

            cv2.putText(frame, "shake head nums: {}".format(TOTAL_FACE), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
