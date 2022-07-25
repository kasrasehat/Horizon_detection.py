from scipy import ndimage
import cv2
import numpy as np
import pickle
#rotation angle in degree
from xml.dom import minidom


def draw_horizon(tan_teta, y_centr, frame):

    # y1=510 , y2= 623 is equal to tan_teta = -0.0587 and teta = -3degree
    # important: origin of y is the highest line and positive direction of y towards down so 510 places higher than 623
    x1 = 0
    y1 = int(np.round(y_centr + tan_teta * 960))
    x2 = 1920
    y2 = int(np.round(y_centr - tan_teta * 960))
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame

def draw_h(frame, y1, y2):

    # y1=510 , y2= 623 is equal to tan_teta = -0.0587 and teta = -3degree
    # important: origin of y is the highest line and positive direction of y towards down so 510 places higher than 623
    x1 = 0
    x2 = frame.shape[1]
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


with open('E:/HD/A/90.xml', 'r') as f:
    data = f.read()

data = data.split(',')
tan_teta = -(int(data[3])-int(data[1]))/1920
y_centr = int(data[5])
b= cv2.imread('E:/HD/A/90.jpg')

#canvas = draw_h(frame= b, y1=int(data[1]), y2=int(data[3]))
canvas = draw_horizon(tan_teta=tan_teta, y_centr= y_centr, frame=b)
cv2.imshow('video', canvas)
cv2.waitKey(5)




