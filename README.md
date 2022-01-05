# horizon-line-detection
first step of this project is converting videos into frames.
afterwards we have to save all of data in a numpy array. 
because of high volume of data we will encounter out of memory error.
for this reason, we can change the strategy of problem solving.
instead of passing each frame into numpy array we can pass the path of each frame to numpy array.
in the training step, data loader is going to load the index of this frames. 
frames related to each index will be loaded and then feed into model in order to train the model.

second solution for this problem is completely different from first one.
In this method there is not learning. we are going to implement edge detecton and hough transform with this assumption that 
horizon is a line. 
##steps of trainable method
###1- video will be read frame by frame
###2- each frame is  sent to a detection function consist of edge detection 
####A- an image with 5 channel including RGB and edge detected channel including angle and magnitude of gradient will be created
####B- FIVE-CHANNEL-IMAGE will be fed into a convolution network
####C- output of this network would be Y, Cos, Sin
###3- horizon line will be drawn on each frame and then frame returned and shown  

##Data Structure
####Dataset which will be fed to dataloader consist of a list including names of frames and a numpy array including parameters corrosponding to each frame