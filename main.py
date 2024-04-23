import cv2 
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('vid.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()