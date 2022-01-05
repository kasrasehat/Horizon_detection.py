import cv2
import numpy as np
import os
import scipy.io
from tqdm import tqdm, trange
import time
from IPython.display import clear_output
from drawing import draw_horizon as dr
from drawing import noisy
from sklearn.model_selection import train_test_split
import re
import pickle
#split data into train and validation not test

path1 = 'Data/frames'
path2 = 'Data/VIS_Onshore/HorizonGT'

files = os.listdir(path1)
inp_frames = np.empty((0, 1080, 1920, 3 ), np.uint8)
inputs = []
outputs = []
for file in tqdm(files):

     time.sleep(0.5)
     horizon_data = scipy.io.loadmat(path2 + '/' + file + '_HorizonGT.mat')
     data = np.transpose(horizon_data['structXML'])
     frame_n1 = horizon_data['structXML'].shape[1]
     data = np.reshape(np.array(data.tolist()), (frame_n1, 4))
     data[:, 0] = data[:, 0] / 1920
     data[:, 1] = data[:, 1] / 1080
     frames = os.listdir(path1 + '/' + file )
     frame_n2 = len(frames)
     sorted_frames = []
     
     for i in range(frame_n2):
          
          for frame in frames:
               
               if int(re.search(r'\d+', frame).group(0)) == i:
                    sorted_frames.append(frame)
                 
     frame_n = min(frame_n1, frame_n2)
     data = data[0:frame_n,:]
     sorted_frames = sorted_frames[:frame_n]
     sorted_frames = [path1+ '/'+ file+ '/'+ name for name in sorted_frames]
     inputs.extend(sorted_frames)
     outputs.extend(data.tolist())


     #for i in tqdm(range(frame_n)):
          #time.sleep(0.3)
          #clear_output(True)
          #path = path1 + '/' + file + '/' + 'frame_{}.jpg'.format(i)
          #frame = cv2.imread(path) / 255
          #if i%8 == 1:
               #noise_type ="gauss"
               #frame = np.uint8(np.clip(noisy(noise_type, frame), 0, 1) * 255)
          #elif i%8 == 3:
               #noise_type = "s&p"
               #frame = np.uint8(np.clip(noisy(noise_type, frame), 0, 1) * 255)
          #elif i%8 == 5:
               #noise_type = "poisson"
               #frame = np.uint8(np.clip(noisy(noise_type, frame), 0, 1) * 255)
          #elif i%8 == 7:
               #noise_type = "speckle"
               #frame = np.uint8(np.clip(noisy(noise_type, frame), 0, 1) * 255)
          #else: frame = frame * 255

          #inputs.append(frame)
          #last_frame = np.reshape(frame, (1, 1080, 1920, 3))
          #inp_frames = np.append(inp_frames, last_frame, axis= 0 )
          #frame = dr(data[0, :], inp_frames[0, :, :, :])


x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.055, shuffle = False)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle = True)

with open("E:/codes_py/horizon_detection/Data/data/x_train.txt", "wb") as fp:   #Pickling
   pickle.dump(x_train, fp)

with open("E:/codes_py/horizon_detection/Data/data/y_train.txt", "wb") as fp:   #Pickling
   pickle.dump(y_train, fp)

with open("E:/codes_py/horizon_detection/Data/data/x_test.txt", "wb") as fp:   #Pickling
   pickle.dump(x_test, fp)

with open("E:/codes_py/horizon_detection/Data/data/y_test.txt", "wb") as fp:   #Pickling
   pickle.dump(y_test, fp)

with open("E:/codes_py/horizon_detection/Data/data/x_valid.txt", "wb") as fp:   #Pickling
   pickle.dump(x_valid, fp)

with open("E:/codes_py/horizon_detection/Data/data/y_valid.txt", "wb") as fp:   #Pickling
   pickle.dump(y_valid, fp)
