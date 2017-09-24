# coding: utf-8

# An example using startStreams
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import os
import math
import random
import imutils
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

sys.path.append('/home/salil/Documents/SSD-Tensorflow/')
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

label_dict = {'none': 0,
     'cup': 1,
     'beer': 2,
     'champagne': 3,
     'deodorant' : 4,
     'gum' : 5,
     'hat' : 6,
     'redwine' : 7,
     'whitewine' : 8,
     'batteries' : 9,
     'altoids' : 10,
     'dryspray' : 11,
     'notebook' : 12,
     'razor' : 13,
     'shirt' : 14,
     'can' : 15,
     'pencils' : 16}
object_labels = {v: k for k, v in label_dict.items()}


slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='Get min Confident lvl to display.')
parser.add_argument('--c', type=float, help='minconfidence needed to display box', default=.25)
parser.add_argument('--s', type=float, help='min select threshold needed to display box', default=.6)
parser.add_argument('--a', type=int, help='minmum_area', default=1)
args = parser.parse_args()
min_conf = args.c
min_select = args.s
min_area = args.a

# In[5]:



# In[6]:

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)

# ## SSD 300 Model
#
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
#
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[7]:

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = '/home/salil/Documents/SSD-Tensorflow/checkpoints/objects/obj_model.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
#
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
#
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[8]:

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def visualize_box(img, rclasses, rscores, rbboxes, depth_matrix):
    obj = []
    for ind, box in enumerate(rbboxes):
        topleft = (int(box[1] * img.shape[1]), int(box[0] * img.shape[0]))
        botright = (int(box[3] * img.shape[1]), int(box[2] * img.shape[0]))
        area = (botright[0] - topleft[0]) * (botright[1] - topleft[1])

        if area > min_area:
            obj.append(img[topleft[1]:botright[1], topleft[0]:botright[0]])
            try:
                depth_matrix[depth_matrix == np.inf] = np.nan
                # depth = int(np.nanmean(depth_matrix[topleft[1]:botright[1], topleft[0]:botright[0]]))

                # print(depth_matrix[int(4*(topleft[1]+botright[1])/10):int(6*(topleft[1]+botright[1])/10),
                #                        int(4*(topleft[0]+botright[0])/10):int(6*(topleft[0]+botright[0])/10)])

                depth = int(np.nanmean(depth_matrix[int(4*(topleft[1]+botright[1])/10):int(6*(topleft[1]+botright[1])/10),
                                       int(4*(topleft[0]+botright[0])/10):int(6*(topleft[0]+botright[0])/10)]))
            except ValueError:

                depth = 'depth is NA'


            img = cv2.rectangle(img, topleft, botright, (0, 255, 0), 1)

            # print(rscores[ind])
            img = cv2.putText(img, str(depth), (topleft[0], topleft[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 255, 55), 1,
                               cv2.LINE_AA)


            # img = cv2.putText(img, object_labels[rclasses[ind]], topleft, cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1,
            #                   cv2.LINE_AA)

            # img = cv2.putText(img, str(rscores[ind]), (topleft[0], botright[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
            #                   cv2.LINE_AA)

    return img, obj


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('head_demo.avi', fourcc, 12.0, (640, 480))

sess = tf.Session()

count = 0
x_axis_rotate = 0

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
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)
color_depth_map = np.zeros((424, 512), np.int32).ravel()
bigdepth = Frame(1920, 1082, 4)
np.set_printoptions(threshold=np.inf)
while True:
    frames = listener.waitForNewFrame()

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]

    if enable_rgb and enable_depth:
        registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth,
                       color_depth_map=color_depth_map)
    # elif enable_depth:
    #     registration.undistortDepth(depth, undistorted)
    # img = cv2.resize(registered.asarray(np.uint8)[:, :, 0:4], (512, 424))

    #cv2.imshow("unrotated", img)

    # bigdepth_img = np.rot90(cv2.resize(bigdepth.asarray(np.float32), (int(1920 / 3), int(1082 / 3))),3)
    # img = np.rot90(cv2.resize(color.asarray()[:, :, 0:3], (int(1920 / 3), int(1080 / 3))),3)
    bigdepth_img = np.rot90(cv2.resize(bigdepth.asarray(np.float32), (int(1920/2), int(1082/2))),3)
    img = np.rot90(cv2.resize(color.asarray()[:, :, 0:3], (int(1920/2), int(1082/2))),3)

    # bigdepth_img = np.rot90(cv2.resize(bigdepth.asarray(np.float32), (int(1920), int(1082))),3)
    # img = np.rot90(cv2.resize(color.asarray()[:, :, 0:3], (int(1920), int(1080))),3)

    #cv2.imshow("bigdepth", bigdepth_img)


    img = img.copy()
    rclasses, rscores, rbboxes = process_image(img, select_threshold=min_select, nms_threshold=min_conf)

    for bbox in rbboxes:
        img.shape[0] ,
    img, cropped_head_list = visualize_box(img, rclasses, rscores, rbboxes, bigdepth_img)

    cv2.imshow("color_img", img)

    # if enable_depth:
    # cv2.imshow("ir", ir.asarray() / 65535.)
    # cv2.imshow("depth", depth.asarray() / 4500.)
    # cv2.imshow("color_depth_map", color_depth_map.reshape(424, 512))
    # print(depth.asarray().shape)
    # cv2.imshow("undistorted", undistorted.asarray(np.float32) / 4500.)
    # if enable_rgb:
    # cv2.imshow("color", img)
    # if enable_rgb and enable_depth:
    # cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)