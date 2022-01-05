import cv2
import  numpy as np
import torch

#params are the output of the network which are y_center, cos and sin of teta
def draw_horizon(params, frame):
    y_cen = params[1] * 1080
    tan_tet = params[2] / params[3]
#y1=510 , y2= 623 is equal to tan_teta = -0.0587 and teta = -3degree
#important: origin of y is the highest line and positive direction of y towards down so 510 places higher than 623    
    x1 = 0
    y1 = int(np.round(y_cen + tan_tet * 960))

    x2 = 1920
    y2 = int(np.round(y_cen - tan_tet * 960))

    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(5000)
    
    return frame


def draw_h(frame, params):
    y_cen = params[0][0, 0].item() * frame.shape[0]
    tan_tet = params[1].item() / params[2].item()
    # y1=510 , y2= 623 is equal to tan_teta = -0.0587 and teta = -3degree
    # important: origin of y is the highest line and positive direction of y towards down so 510 places higher than 623    
    x1 = 0
    y1 = int(np.round(y_cen + (tan_tet * frame.shape[1] / 2)))

    x2 = frame.shape[1]
    y2 = int(np.round(y_cen - (tan_tet * frame.shape[1] / 2)))

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame





def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
      
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy
    