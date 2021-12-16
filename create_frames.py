import cv2
import os

path1 = 'Data/VIS_Onshore/Videos'
path2 = 'Data/VIS_Onshore/HorizonGT'

files = os.listdir(path1)
horizon_data = os.listdir(path2)
file = [value[:-14] for value in horizon_data]
for video in files:
    if video[:-4] in file:
        capture = cv2.VideoCapture(path1 + '/' + video)
        frameNr = 0
        fps = capture.get(cv2.CAP_PROP_FPS)
        amountOfFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT )
        store_path = 'Data/frames/' +  video[:-4]
        os.makedirs(store_path)

        while (True):

            success, frame = capture.read()

            if success:
                cv2.imwrite('{}/frame_{}.jpg'.format(store_path, frameNr), frame)

            else:
                break

            frameNr = frameNr + 1

capture.release()