from utils.camera.kinect import KinectCamera
from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np

#(score, diff) = compare_ssim(grayA, grayB, full=True)
kinect = KinectCamera(1)
original = np.rot90(cv2.resize(kinect.get_frames()[0], (1920 // 2, 1080 // 2)), 3)
#original = cv2.GaussianBlur(original,(5,5),0)

threshold = 30
kernel = np.ones((5,5), np.uint8)
while True:
    rgb = np.rot90(cv2.resize(kinect.get_frames()[0],(1920//2,1080//2)),3)
    #rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    diff = cv2.absdiff(original , rgb)

    bw_diff = np.sum(diff,axis = 2)

    bw_diff[bw_diff < threshold] = 0
    bw_diff[bw_diff >= threshold] = 255
    bw_diff = cv2.GaussianBlur(bw_diff.astype(np.uint8), (5, 5), 0)
    bw_diff = cv2.dilate(bw_diff, kernel, iterations=1)
    print(bw_diff)
    #(score, diff) = compare_ssim(grayA, grayB, full=True)
    cv2.imshow("difference", bw_diff.astype(np.uint8))
    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break