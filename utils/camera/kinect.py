from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
import sys
import numpy as np
import cv2
import time
from pylibfreenect2 import createConsoleLogger, setGlobalLogger


setGlobalLogger(None)

class KinectCamera(object):

    def __init__(self, camera):
        try:
            from pylibfreenect2 import OpenCLPacketPipeline

            pipeline = OpenCLPacketPipeline()
        except:
            try:
                from pylibfreenect2 import OpenGLPacketPipeline

                pipeline = OpenGLPacketPipeline()
            except:
                from pylibfreenect2 import CpuPacketPipeline

                pipeline = CpuPacketPipeline()
        print("Packet pipeline:", type(pipeline).__name__)

        self.fn = Freenect2()

        num_devices = self.fn.enumerateDevices()

        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)
        else:
            print('print device found')

        self.serial = self.fn.getDeviceSerialNumber(camera)
        self.device = self.fn.openDevice(self.serial, pipeline=pipeline)
        self.camera = camera
        self._configure_camera_parameters()

        types = 0
        self.enable_rgb = True
        self.enable_depth = True
        if self.enable_rgb:
            types |= FrameType.Color
        if self.enable_depth:
            types |= (FrameType.Ir | FrameType.Depth)
        self.listener = SyncMultiFrameListener(types)

        # Register listeners
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)


        if self.enable_rgb and self.enable_depth:
            self.device.start()
        else:
            self.device.startStreams(rgb=self.enable_rgb, depth=self.enable_depth)

        # NOTE: must be called after device.start()
        if self.enable_depth:
            self.registration = Registration(self.device.getIrCameraParams(),
                                        self.device.getColorCameraParams())

        self.undistorted = Frame(512, 424, 4)
        self.registered = Frame(512, 424, 4)
        self.color_depth_map = np.zeros((424, 512), np.int32).ravel()
        self.bigdepth = Frame(1920, 1082, 4)


    def _configure_camera_parameters(self):
        if self.camera == 1:
            print('configure camera 1')

    def get_frames(self):

        self.frames = self.listener.waitForNewFrame()
        if self.enable_rgb:
            color = self.frames["color"]
        if self.enable_depth:
            ir = self.frames["ir"]
            depth = self.frames["depth"]

        if self.enable_rgb and self.enable_depth:
            self.registration.apply(color, depth, self.undistorted, self.registered, bigdepth=self.bigdepth,
                           color_depth_map=self.color_depth_map)

        # bigdepth_img = self.bigdepth.asarray(np.float32)
        # img = color.asarray()[:, :, 0:3] #wtf, this deoesnt' work but need a cv2 resize???
        bigdepth_img = cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 1), int(1082 / 1)))
        img = cv2.resize(color.asarray()[:, :, 0:3], (int(1920 / 1), int(1080 / 1)))

        self.listener.release(self.frames)
        return img, bigdepth_img

    def __del__(self):
        self.device.stop()
        self.device.close()

if __name__ == '__main__':
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('object_no_light.avi', fourcc, 30.0, (int(1080/3), int(1920 / 3)))
    kinect_right=  KinectCamera(1)
    kinect_left = KinectCamera(0)
    previouscounter = 0
    frameinterval = 7
    counter =0
    while True:
        counter +=1
        #out.write(np.fliplr(kinect_left.get_frames()[0]))
        time.sleep(1)
        left = kinect_left.get_frames()[0]
        right = kinect_right.get_frames()[0]
        left = left.copy()
        # left = cv2.resize(np.rot90(left,3), (int(1080/2), int(1920/2)))
        # imghalves = [left[:542, :].copy(), left[418:, :].copy()]
        if counter%frameinterval== 0:
            # cv2.imwrite('truetop' + str(previouscounter + counter // frameinterval) + '.png', imghalves[0])
            # cv2.imwrite('truebot' + str(previouscounter + counter // frameinterval) + '.png', imghalves[1])

            cv2.imwrite('dual_left_'+str(previouscounter +counter//frameinterval)+'.png', left)
            cv2.imwrite('dual_right_' + str(previouscounter +counter//frameinterval)+'.png', right)




        left = cv2.putText(left, str(counter), (int(105), int(155)), cv2.FONT_HERSHEY_SIMPLEX, 6, (155, 255, 55), 10, cv2.LINE_AA)



        right = right.copy()
        right = cv2.resize(np.rot90(right,1), (int(1080/2), int(1920/2)))
        left = cv2.resize(np.rot90(left, 3), (int(1080 / 2), int(1920 / 2)))
        # cv2.imshow('top', imghalves[0])
        # cv2.imshow('bot', imghalves[1])
        cv2.imshow('left', left)
        cv2.imshow('right', right)



        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            #kinect_right.__del__()
            kinect_left.__del__()
            break

    #out.release()
    sys.exit(0)


