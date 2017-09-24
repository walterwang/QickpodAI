from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
import sys
import numpy as np
import cv2
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


        bigdepth_img = np.rot90(cv2.resize(self.bigdepth.asarray(np.float32), (int(1920 / 3), int(1082 / 3))),3)
        img = np.rot90(cv2.resize(color.asarray()[:, :, 0:3], (int(1920 / 3), int(1080 / 3))),3)

        self.listener.release(self.frames)
        return img, bigdepth_img

    def __del__(self):
        self.device.stop()
        self.device.close()
if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('example_recording_for_salil.avi', fourcc, 20.0, (int(1080/3), int(1920 / 3)))
    kinect_right=  KinectCamera(1)
    kinect_left = KinectCamera(0)
    while True:
        out.write(np.rot90(kinect_right.get_frames()[0],2))
        # cv2.imshow('left', kinect_left.get_frames()[0])
        # cv2.imshow('right', np.rot90(kinect_right.get_frames()[0],2))
        key = cv2.waitKey(delay=1)

        if key == ord('q'):
            break
    out.release()
    sys.exit(0)


