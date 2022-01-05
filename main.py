import cv2
import torch
from networks import CNNmodel1 as Net
import torch.nn.functional as F
from canny_filter1 import edge_channel
import numpy as np
from drawing import draw_h
from collections import deque
from imutils.video import FPS


if __name__ == "__main__" :

    device = torch.device("cuda:0")
    model = Net().to(device)
    myload = torch.load("weights/CNNmodel.pt")
    try:
        model.load_state_dict(myload['state_dict'])
    except:
        model.load_state_dict(myload)

    path_vid = 'E:/codes_py/horizon_detection/Data/VIS_Onshore/Videos/MVI_1645_VIS.avi'
    frame_n = 0
    queue = deque(maxlen = 20)
    queue1 = deque(maxlen=20)
    video_capture = cv2.VideoCapture(path_vid)
    fps = FPS().start()
    success = True
    

    while True:
        success, frame_org = video_capture.read()
        if success :

            grad, angle = edge_channel(frame_org)
            edge_features = np.append(np.reshape(grad, (108, 192, 1)), np.reshape(angle, (108, 192, 1)), axis=2)
            frame = cv2.resize(frame_org, (192, 108))
            frame = np.reshape(np.append(frame, edge_features, axis=2), (5, 108, 192))
            params = model(torch.tensor(frame/255).unsqueeze(0).to(device, dtype=torch.float))
            params[2] = (2 * params[2]) - 1
            queue.append(params[0][0, 0])
            queue1.append(params[1])
            params[0][0, 0] = sum(queue)/len(queue)
            params[1] = sum(queue1)/len(queue1)
            canvas = draw_h(frame_org, params)
            cv2.imshow('video', canvas)
            frame_n += 1
            fps.update()
        else:
            print(frame_n)
            video_capture.release()
            cv2.destroyWindow('video')
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    fps.stop()
    video_capture.release()
    cv2.destroyWindow('video')
    print("elapsed time: {:.2f} milisecond".format(fps.elapsed()))
    print("fps: {:.2f} frame/second".format(fps.fps()))