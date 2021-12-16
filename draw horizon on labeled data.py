import cv2
import scipy.io

def draw_hor(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        gray_roi = gray[y:y+h ,x:x+w]
        frame_roi = frame[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        gray_roi = gray[y:y+h ,x:x+w]
        frame_roi = frame[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 5)
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame

path = 'Data/VIS_Onshore/HorizonGT/MVI_1469_VIS_HorizonGT.mat'
path_vid = 'Data/VIS_Onshore/Videos/MVI_1469_VIS.avi'
frame_n = 0
horizon_cor = mat = scipy.io.loadmat(path)
video_capture = cv2.VideoCapture(path_vid)
while True:
    _ ,frame = video_capture.read()
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
    canvas = draw_hor(gray, frame )
    cv2.imshow('video', canvas)
    if cv2.waitKey(1.5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyWindow()