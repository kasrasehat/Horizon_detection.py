import cv2
import torch
from networks import CNNmodel1 as Net
import torch.nn.functional as F
from canny_filter import edge_channel
import numpy as np
from drawing import draw_h
from imutils.video import FPS




if __name__ == "__main__" :

    path_vid = 'E:/codes_py/horizon_detection/Data/VIS_Onshore/Videos/MVI_1481_VIS.avi'
    #'E:/codes_py/horizon_detection/Data/VIS_Onshore/Videos/MVI_1482_VIS.avi'
    frame_n = 0
    video_capture = cv2.VideoCapture(path_vid)
    fps = FPS().start()
    success = True

    while True:
        success, frame = video_capture.read()
        if success:
            grad, angle = edge_channel(frame)
            lines = cv2.HoughLines(grad, 1, np.pi / 160, 200)
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                tan_teta = (y2 - y1) / (x2 - x1)
                x11 = 0
                x22 = frame.shape[1]
                y11 = y1 - int(tan_teta * (x1))
                y22 = y1 + int(tan_teta * (x22 - x1))

                cv2.line(frame, (x11, y11), (x22, y22), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            fps.update()
        else:
            video_capture.release()
            cv2.destroyWindow('video')
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    fps.stop()
    video_capture.release()
    cv2.destroyWindow('video')
    print("elapsed time: {:.2f} frame/second".format(fps.elapsed()))