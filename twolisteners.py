from utils.camera.kinect import KinectCamera
from threading import Thread
import cv2
import time

from pylibfreenect2 import createConsoleLogger, setGlobalLogger


setGlobalLogger(None)

def get_frames(waittime):

    while True:

        #cv2.imshow(name, demoai.get_frames()[0])
        print(demoai.get_frames()[0].shape)
        time.sleep(waittime)
        key = cv2.waitKey(delay = 1)
        if key == ord('q'):
            break


if __name__=='__main__':
    demoai = KinectCamera(0)
    Thread(target=get_frames, args=(0,)).start()
    Thread(target=get_frames, args=(1,)).start()

