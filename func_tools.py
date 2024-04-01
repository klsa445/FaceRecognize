#coding:utf-8
import cv2
from imutils import face_utils
import numpy as np
from PyQt5.QtGui import QPixmap,QImage

def cvimg_to_qpiximg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qpix_img = QPixmap(qimg)
    return qpix_img

def draw_face_contour(image, shape):
    # 获取左右眼的索引起始值
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
    # 嘴的索引
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]
    jaw = shape[jStart:jEnd]
    mouth = shape[mStart:mEnd]

    # 寻找左右眼睛凸包
    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)

    # 寻找凸包
    mouth_hull = cv2.convexHull(mouth)

    # 画出嘴凸包区域
    cv2.drawContours(image, [mouth_hull], -1, (255, 255, 255), 2)

    # 画出眼睛凸包区域
    cv2.drawContours(image,[left_eye_hull],-1,(255, 255, 255), 2)
    cv2.drawContours(image,[right_eye_hull],-1,(255, 255, 255), 2)
    # 画脸颊线
    cv2.polylines(image, [jaw], False, (255,255,255), 2)  # 原始少量散点构成的折线图
    # 画鼻子线
    cv2.polylines(image, [nose[:4]], False, (255, 255, 255), 2)
    cv2.polylines(image, [nose[4:]], False, (255, 255, 255), 2)
    return image

def MAR(mouth):
    # 默认二范数：求特征值，然后求最大特征值得算术平方根
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59（人脸68个关键点）
    B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    return (A + B) / (2.0 * C)

def EAR(eye):
    # 默认二范数：求特征值，然后求最大特征值得算术平方根
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def SHAKEAR(nose, jaw):
    # 计算鼻子上一点"27"到左右脸边界的欧式距离
    face_left1 = np.linalg.norm(nose[0] - jaw[0])  # 27, 0
    face_right1 = np.linalg.norm(nose[0] - jaw[16])  # 27, 16
    # 计算鼻子上一点"30"到左右脸边界的欧式距离
    face_left2 = np.linalg.norm(nose[3] - jaw[2])  # 30, 2
    face_right2 = np.linalg.norm(nose[3] - jaw[14])  # 30, 14

    # 创建元组，用以保存4个欧式距离值
    face_distance = (face_left1, face_right1, face_left2, face_right2)

    return face_distance

def NodAR(nose, jaw):
    # 鼻子长度
    nose_dist = np.linalg.norm(nose[0] - nose[3])
    # 脸颊宽度
    face_dist = np.linalg.norm(jaw[3] - jaw[13])
    far = nose_dist / face_dist
    return far