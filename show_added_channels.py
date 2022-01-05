import cv2
import  numpy as np
import os
from canny_filter import edge_channel
import scipy.io


path = "E:/codes_py/horizon_detection/Data/frames/MVI_1471_VIS/frame_11.jpg"
#path1 = 'Data/VIS_Onshore/HorizonGT/MVI_1619_VIS_HorizonGT.mat'
frame = cv2.imread(path)
#frame = cv2.bilateralFilter(frame, 5, 75, 75)
#frame = cv2.resize(frame, (192, 108), interpolation= cv2.INTER_AREA)
#frame_n = 0
#horizon_cor =  scipy.io.loadmat(path1)
#params = horizon_cor['structXML']
#canvas = draw_hor(frame, frame_n, params)
#frame = cv2.resize(canvas, (1920, 1080),interpolation=cv2.INTER_LINEAR)
#cv2.imshow('angle', frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


grad, angle = edge_channel(frame)
#lines = cv2.HoughLinesP(grad, 1, np.pi/180, 100, 50)
lines = cv2.HoughLines(grad, 1, np.pi/360, 100)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    tan_teta = (y2 - y1)/(x2 - x1)
    x11 = 0
    x22 = frame.shape[1]
    y11 = y1 - int(tan_teta *(x1))
    y22 = y1 + int(tan_teta * (x22 - x1))
    cv2.line(frame,(x11,y11),(x22,y22),(0, 255, 0),2)
#for x1,y1,x2,y2 in lines[0]:
  #  tan_teta = (y2 - y1) / (x2 - x1)
  #  x11 = 0
  # x22 = frame.shape[1]
  #  y11 = y1 - int(tan_teta *(x1))
  #  y22 = y1 + int(tan_teta * (x22 - x1))
  #  cv2.line(frame,(x11,y11),(x22,y22),(0,255),2)



cv2.imshow('frame', frame)
cv2.imshow('grad', grad)
cv2.imshow('angle', angle)
cv2.waitKey(0)
cv2.destroyAllWindows()
