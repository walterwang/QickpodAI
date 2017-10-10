from utils.camera.kinect import KinectCamera
import cv2
import sys


r = KinectCamera(1)

while True:
    rgb = r.get_frames()[0]
    cv2.imshow('image', rgb)
    key = cv2.waitKey(delay=1)

    if key == ord('q'):
        r.__del__()
        break


sys.exit(0)