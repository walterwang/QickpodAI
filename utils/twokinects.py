import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
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

enable_rgb = True
enable_depth = True

fn = Freenect2()
fn1 = Freenect2()
# num_devices = fn.enumerateDevices()
#
# if num_devices == 0:
#     print("No device connected!")
#     sys.exit(1)

serial1 = fn.getDeviceSerialNumber(0)
serial2 = fn1.getDeviceSerialNumber(1)
device1 = fn.openDevice(serial1, pipeline=pipeline)
device2 = fn1.openDevice(serial2, pipeline=pipeline)

types = 0
types |= FrameType.Color
listener = SyncMultiFrameListener(types)
device1.setColorFrameListener(listener)

device1.start()

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
color_depth_map = np.zeros((424, 512), np.int32).ravel()
bigdepth = Frame(1920, 1082, 4)
np.set_printoptions(threshold=np.inf)

while True:
    frames = listener.waitForNewFrame()


    color = frames["color"]

    cv2.imshow("color", cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3))))


    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device1.stop()
device1.close()

sys.exit(0)
