from multiprocessing import Process, Queue
from utils.camera.kinect import KinectCamera
import cv2
import time


def writer(queue):
    
    demokinect = KinectCamera(0)

    while True:
        frame = demokinect.get_frames()[0]

        queue.put(frame)
        

def reader(queue):
    while True:
        frame = queue.get()
        time.sleep(.2)
        cv2.imshow('frames', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
            
if __name__ == '__main__':
    queue = Queue()
    reader_p = Process(target = reader, args=((queue), ))
    #reader_p.daemon = True
    reader_p.start()

    _start = time.time()
    writer(queue)
    reader_p.join()