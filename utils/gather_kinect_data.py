# coding: utf-8

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
import argparse

parser = argparse.ArgumentParser(description='make videos')
parser.add_argument('--c', type=float, help='minconfidence needed to display box', default=.25)
parser.add_argument('--s', type=float, help='min select threshold needed to display box', default=.6)
parser.add_argument('--a', type=int, help='minmum_area', default=1)
parser.add_argument('--v', type=bool, help='make video', default=False)
args = parser.parse_args()
min_conf = args.c
min_select = args.s
min_area = args.a
make_video = args.v
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

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

listener = SyncMultiFrameListener(
    FrameType.Color | FrameType.Ir | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

# Optinal parameters for registration
# set True if you need
need_bigdepth = False
need_color_depth_map = False
counter = 0
bigdepth = Frame(1920, 1082, 4) if need_bigdepth else None
color_depth_map = np.zeros((424, 512),  np.int32).ravel() \
    if need_color_depth_map else None

if make_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_data/object_detection1.avi',fourcc, 20.0, (int(1080 / 2),int(1920 / 2)))

while True:
    frames = listener.waitForNewFrame()

    color = frames["color"]
    ir = frames["ir"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered,
                       bigdepth=bigdepth,
                       color_depth_map=color_depth_map)

    # NOTE for visualization:
    # cv2.imshow without OpenGL backend seems to be quite slow to draw all
    # things below. Try commenting out some imshow if you don't have a fast
    # visualization backend.
    #cv2.imshow("ir", ir.asarray() / 65535.)
    #cv2.imshow("depth", depth.asarray() / 4500.)
    img = np.fliplr(np.rot90(cv2.resize(color.asarray()[:,:,0:3],(int(1920 / 2), int(1080 / 2))), 3))

    cv2.imshow("color", img)
    #cv2.imshow("registered", registered.asarray(np.uint8))
    #
    # if need_bigdepth:
    #     cv2.imshow("bigdepth", cv2.resize(bigdepth.asarray(np.float32),
    #                                       (int(1920 / 3), int(1082 / 3))))
    # if need_color_depth_map:
    #     cv2.imshow("color_depth_map", color_depth_map.reshape(424, 512))
    if make_video:
        out.write(img)
    else:
        counter+=1
        cv2.imwrite('training_img%s.png'%str(counter).zfill(5), img)
        cv2.waitKey(1000)

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

out.release()
device.stop()
device.close()

sys.exit(0)
