# coding:utf-8
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np


def NodAR(nose, jaw):
    # 鼻子长度
    nose_dist = np.linalg.norm(nose[0] - nose[3])
    # 脸颊宽度
    face_dist = np.linalg.norm(jaw[3] - jaw[13])
    far = nose_dist / face_dist
    return far

def main():
    FAR_THRESH = 0.33  # 点头阈值
    # 初始化
    COUNTER_NOD = 0
    TOTAL_NOD = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']

    vs = VideoStream(src=0).start()
    time.sleep(1.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 在灰度框中检测人脸
        rects = detector(gray, 0)

        # 进入循环
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 提取鼻子和下巴的坐标，然后使用该坐标计算鼻子到左右脸边界的欧式距离
            nose = shape[nStart:nEnd]
            jaw = shape[jStart:jEnd]
            NOSE_JAW_Distance = NodAR(nose, jaw)
            print(NOSE_JAW_Distance)

            if NOSE_JAW_Distance > FAR_THRESH:
                COUNTER_NOD += 1

            else:
                # 如果点头帧计数器不等于0，则增加张嘴的总次数
                if COUNTER_NOD >= 2:
                    TOTAL_NOD += 1
                COUNTER_NOD = 0

            # 画出点头次数
            cv2.putText(frame, "nod head nums: {}".format(TOTAL_NOD), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
