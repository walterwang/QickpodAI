from utils.camera.kinect import KinectCamera
import cv2
import numpy as np
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
import time


setGlobalLogger(None)


left = KinectCamera(0)
#right = KinectCamera(1)


start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0
fps =0


original =left.get_frames()[0]
# original[original == np.inf] = 0
# original = np.nan_to_num(original)
# print(np.amax(original))

while True:
    # counter +=1
    # if (time.time() - start_time) > x :
    #     fps = counter / (time.time() - start_time)
    #     counter = 0
    #     start_time = time.time()
    img, depth_img = left.get_frames()
    depth_img[depth_img ==np.inf] =0
    depth_img = np.nan_to_num(depth_img)

    #current_frame = depth_img
    current_frame = img.copy()
    #current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # cv2.putText(img, str(fps), (int(105), int(155)), cv2.FONT_HERSHEY_SIMPLEX, 6, (155, 255, 55), 10,
    #                cv2.LINE_AA)
    #current_frame=np.nan_to_num(current_frame)

    frame_diff = cv2.absdiff(current_frame, original)


    frame_diff[np.where((frame_diff > [10, 10, 10]).all(axis=2))] = [255, 255, 255]
    frame_diff = cv2.medianBlur(frame_diff, 5)
    #frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

    #ret1, frame_diff = cv2.threshold(frame_diff, 14, 255, cv2.THRESH_BINARY)

    #_, contours, hierarchy = cv2.findContours(frame_diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # approx = []
    # for c in contours:
    #     epsilon = 0.01 * cv2.arcLength(c, True)
    #     approx.append(cv2.approxPolyDP(c, epsilon, True))

    # detector = cv2.SimpleBlobDetector_create()
    # keypoints = detector.detect(frame_diff)
    # blob_image = cv2.drawKeypoints(frame_diff, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.drawContours(frame_diff, contours, -1, (0, 255, 0), 1)

    current_frame = np.fliplr(np.rot90(cv2.resize(frame_diff, (1080//2,1920//2)),2))
    cv2.imshow('left', current_frame)
    key = cv2.waitKey(delay=1)

    if key == ord('q'):
        cv2.destroyAllWindows()
        break