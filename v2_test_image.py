import numpy as np
from v2_horizon_detection_class import detect_horizon
import cv2

def draw_horizon(frame, y_centr, teta):

    tan_tet = np.tan(teta)

    x1 = 0
    y1 = int(y_centr + tan_tet * 960)

    x2 = 1920
    y2 = int(y_centr - tan_tet * 960)

    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


detector = detect_horizon()
image = cv2.imread('E:/HD/A/514.jpg')
y_centr, teta = detector.detect(input=image)
frame = draw_horizon(image, y_centr.to('cpu'), teta.to('cpu'))
cv2.imshow('video', frame)
cv2.waitKey(5)