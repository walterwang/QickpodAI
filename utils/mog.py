import numpy as np
import cv2
from utils.camera.kinect import KinectCamera

left = KinectCamera(0)


fgbg = cv2.createBackgroundSubtractorKNN()
while(1):
    frame, depth = left.get_frames()

    depth = np.nan_to_num(depth)
    depth[depth == np.inf] = 0
    #frame = depth



    frame = np.fliplr(np.rot90(cv2.resize(frame, (1080//2, 1920//2)),2))
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
