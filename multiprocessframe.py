from multiprocessing import Process, Queue
from utils.camera.kinect import KinectCamera
import cv2


def writer(queue):
    
    demokinect = KinectCamera(0)

    while True:
        frame = demokinect.get_frames()[0]
        queue.put(frame)
        

def reader(queue):
    while True:
        frame = queue.get()
        cv2.imshow('frames', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
if __name__ == '__main__':
    pass