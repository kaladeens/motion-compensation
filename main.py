import cv2 
import numpy as np
from matplotlib import pyplot as plt

def get_two_frames(video,time):
    #load in video
    cap = cv2.VideoCapture(video)
    #get frame1 from video
    _ , frame1 = cap.read()
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB) #change to RGB

    # Skip frames for time in seconds
    frame_rate = cap.get(cv2.CAP_PROP_FPS) #retrieve framerate
    skip_frames = int(frame_rate * time)
    for _ in range(skip_frames):
        _ , frame2 = cap.read()
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)

    return frame1,frame2

frame1, frame2 = get_two_frames('vid.mp4',.3)
plt.subplot(2,1,1)
plt.imshow(frame1)


plt.subplot(2,1,2)
plt.imshow(frame2)
plt.show()



def find_matches(frame1,frame2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame1,None)
    kp2, des2 = orb.detectAndCompute(frame2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)    

    plt.figure(figsize=(15,5))

    img3 = cv2.drawMatches(frame1,kp1,frame2,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
