import cv2
import scipy.io
#from numba import jit, cuda

#@jit("int32(int32, int32)",target ="CUDA")
def draw_hor(frame, frame_n, params):
    y_cen = params[0 ,frame_n][1]
    tan_tet = params[0 ,frame_n][2]/params[0 ,frame_n][3]

    x1 = 0
    y1 = int(y_cen + tan_tet * 960)

    x2 = 1920
    y2 = int(y_cen - tan_tet * 960)

    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame

if __name__ == "__main__" :
    path = 'Data/VIS_Onshore/HorizonGT/MVI_1584_VIS_HorizonGT.mat'
    path_vid = 'Data/VIS_Onshore/Videos/MVI_1584_VIS.avi'
    frame_n = 0
    horizon_cor =  scipy.io.loadmat(path)
    params = horizon_cor['structXML']
    video_capture = cv2.VideoCapture(path_vid)
    success = True
    while True:
        success, frame = video_capture.read()
        if success :
            canvas = draw_hor(frame, frame_n, params)
            cv2.imshow('video', canvas)
            frame_n += 1
        else:
            video_capture.release()
            cv2.destroyWindow('video')
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyWindow('video')