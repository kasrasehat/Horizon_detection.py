# horizon-line-detection
first step of this project is converting videos into frames.
afterwards we have to save all of the data in a numpy array. 
because of high volume of data we will encounter out of memory error.
for this reason, we can change the strategy of problem solving.
instead of passing each frame into numpy array we can pass the path of each frame to numpy array.
in the training step, data loader is going to load the index of this frames. 
frames related to each index will be loaded and then feed into model in order to train the model.

second solution for this problem is completely different from first one.
In this method there is not learning. we are going to implement edge detecton and hough transform with this assumption that 
horizon is a line. 
## steps of trainable method
### 1- video will be read frame by frame
### 2- each frame is  sent to a detection function consist of edge detection 
#### A- an image with 5 channel including RGB and edge detected channel including angle and magnitude of gradient will be created
#### B- FIVE-CHANNEL-IMAGE will be fed into a convolution network
#### C- output of this network would be Y, Cos, Sin
### 3- horizon line will be drawn on each frame and then frame returned and shown  

## Data Structure
#### Dataset which will be fed to dataloader consist of a list including names of frames and a numpy array including parameters corrosponding to each frame

## Version 2 for solving horizon line problem:
because of the complexity of input images and also large derivatives in consecutive frames, simple networks like ones used in
V.1 are not useful. Therefor networks with complex structure which have data extractor will be used. VGG16, ResNet or even BEIT seem to have 
good accuracy in such problems.

## Steps for creating version 2:
### Data augmentation:
Because of low number of data we need to augment exist data. augmentation includes rotation and adding noise to each data. 
If rotation method is imposed to image, label is also should be changed. Therefor we can augment 1000 data to more than 10000 data.

### Finding best pretrained feature extractor

### Labeling data in three forms:
#### Method one:
middle point cordinate and line angle

#### Method two:
many points on the line

#### Method four:
first, middle and end point

saved_models file contains weights related to different models and way of implementation of train process.
#### CNNmodel1:
It is related to VGGmodel1 which the last 4 layers were trained and achieve the loss of y equal to 0.00098 and the loss of teta equal to 0.000040.
batch size is set to 16 and lr is 0.01 at the begining of train process 
