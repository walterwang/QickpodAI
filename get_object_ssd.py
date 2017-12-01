# coding: utf-8

# An example using startStreams
import numpy as np
import tensorflow as tf
import cv2
import argparse
import sys

with open("/home/salil/QickpodAI/ssd_path.txt") as f:
    ssd_path=f.readlines()
sys.path.append(ssd_path[0].strip())

from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

label_dict = {'none': 0,
     'mk_brown_wrislet': 1,
     'LMK Messenger Tote Bags 285': 2,
     'sm_peach_backpack': 3,
     'Nine West Reana' : 4,
     'Pahajim women handbag (wine red)' : 5,
     'meixuan_brown_handbag' : 6,
     'sm_bclarre_blush_crossbody' : 7,
     'Steve madden Bdrew Grey Kolt Updated' : 8,
     'Handbag (white)' : 9,
     'black_plain_bag' : 10,
     'black_backpack' : 11,
     'black_ameligalanti' : 12,
     'ghost' : 13,
     'ghost' : 14,
     'ghost' : 15,
    'ghost' : 16,
    'ghost' : 17,
    'ghost' : 18,
    'ghost' : 19,
    'ghost' : 20,
    'ghost' : 21

}
object_labels = {v: k for k, v in label_dict.items()}


slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='Get min Confident lvl to display.')
parser.add_argument('--c', type=float, help='minconfidence needed to display box', default=0)
parser.add_argument('--s', type=float, help='min select threshold needed to display box', default=.3)
parser.add_argument('--a', type=int, help='minmum_area', default=1)
args = parser.parse_args()
min_conf = args.c
min_select = args.s
min_area = args.a


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.Session(config=config)


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

ckpt_filename = ssd_path[1].strip()

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)




def process_image(img, select_threshold=0.5, nms_threshold=0, net_shape=(512, 512)):
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
    #convert to abs cooordinates

    for box in rbboxes:
        box[0]=int(box[0]*img.shape[0])
        box[2]=int(box[2]*img.shape[0])
        box[1]=int(box[1]*img.shape[1])
        box[3]=int(box[3]*img.shape[1])
    rbboxes = rbboxes.astype(int)



    return rclasses, rscores, rbboxes


def visualize_box(img, rclasses, rscores, rbboxes, depth_matrix):
    obj = []
    for ind, box in enumerate(rbboxes):
        # topleft = (int(box[1] * img.shape[1]), int(box[0] * img.shape[0]))
        # botright = (int(box[3] * img.shape[1]), int(box[2] * img.shape[0]))

        topleft = (box[1], box[0])
        botright = (box[3], box[2])
        area = (botright[0] - topleft[0]) * (botright[1] - topleft[1])

        if area > min_area:

            obj.append(img[topleft[1]:botright[1], topleft[0]:botright[0]])
            try:
                depth_matrix[depth_matrix == np.inf] = np.nan

                depth = int(np.nanmean(depth_matrix[int(4*(topleft[1]+botright[1])/10):int(6*(topleft[1]+botright[1])/10),
                                       int(4*(topleft[0]+botright[0])/10):int(6*(topleft[0]+botright[0])/10)]))
            except ValueError:

                depth = 'depth is NA'
            img = cv2.rectangle(img, topleft, botright, (0, 255, 0), 1)

            # print(rscores[ind])
            img = cv2.putText(img, str(depth), (topleft[0], topleft[1]-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (155, 255, 55), 1,
                               cv2.LINE_AA)


            img = cv2.putText(img, object_labels[rclasses[ind]], topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                              cv2.LINE_AA)

            img = cv2.putText(img, str(rscores[ind]), (topleft[0], botright[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                              cv2.LINE_AA)

    return img, obj


sess = tf.Session()


def is_overlapping(obj1, obj2):
    overlap_area = 0
    small_area = obj2[2] * obj2[3]

    obj1 = obj1[:]
    obj1[2] = obj1[0] + obj1[2]
    obj1[3] = obj1[1] + obj1[3]
    obj2 = obj2[:]
    obj2[2] = obj2[0] + obj2[2]
    obj2[3] = obj2[1] + obj2[3]

    dx = min(obj1[2], obj2[2]) - max(obj1[0], obj2[0])
    dy = min(obj1[3], obj2[3]) - max(obj1[1], obj2[1])
    if (dx >= 0) and (dy >= 0):
        overlap_area = dx * dy

    if overlap_area / small_area > 0.5:
        return True
    else:
        return False


def remove_overlapping_objects(objects):
    # Create a graph whose edge represent the two nodes overlap
    selected_objects = []
    overlap_graph = dict()
    for i in range(len(objects)):
        overlap_graph[i] = []

    # Iterate through pairs of hands
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            area_i = objects[i][2] * objects[i][3]
            area_j = objects[j][2] * objects[j][3]
            if area_i > area_j and is_overlapping(objects[i], objects[j]) is True:
                overlap_graph[i].append(j)
                overlap_graph[j].append(i)
            elif area_i < area_j and is_overlapping(objects[j], objects[i]) is True:
                overlap_graph[i].append(j)
                overlap_graph[j].append(i)

    # Among overlapping objects, get the object with highest confidence
    removed_boxes = []
    for i in range(len(objects)):
        confidence = objects[i][4]
        lost = 0
        for j in overlap_graph[i]:
            if confidence < objects[j][4]:
                lost = 1
        if lost == 1:
            removed_boxes.append(i)
    return removed_boxes


def get_bboxes(shape, classes, scores, bboxes):
    objects = []

    for i in range(len(bboxes)):
        obj = [0, 0, 0, 0, 0, 0]
        bbox = bboxes[i]
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        obj[1], obj[0] = p1[0], p1[1]
        obj[3], obj[2] = p2[0] - p1[0], p2[1] - p1[1]
        obj[4] = scores[i]
        obj[5] = classes[i]
        objects.append(obj)

    return objects

def get_objects(img, bigdepth_img):

    imghalves = [img[:542,:].copy(), img[418:,:].copy()]
    imgdepths = [bigdepth_img[:542, :].copy(), bigdepth_img[418:, :].copy()]

    rclasses, rscores, rbboxes = process_image(imghalves[0], select_threshold=0.85)

    rclasses1, rscores1, rbboxes1 = process_image(imghalves[1], select_threshold=0.85)

    rbboxes = rbboxes.tolist()
    rclasses = rclasses.tolist()
    rscores = rscores.tolist()
    rclasses1 = rclasses1.tolist()
    rscores1 = rscores1.tolist()
    rbboxes1 = rbboxes1.tolist()

    final_obj =[]
    final_rclasses=[]
    final_rscores = []
    final_rnames = []
    for m, obj in enumerate(rbboxes):
        if obj[0]<460:
            final_obj.append(obj)
            final_rclasses.append(rclasses[m])
            final_rscores.append(rscores[m])
            final_rnames.append(object_labels[rclasses[m]])

    for i, obj in enumerate(rbboxes1):
        if obj[0] > 10:
            obj[0] = obj[0] + 418
            obj[2] = obj[2] + 418
            final_obj.append(obj)
            final_rclasses.append(rclasses1[i])
            final_rscores.append(rscores1[i])
            final_rnames.append(object_labels[rclasses1[i]])

    img =img.copy()
    objects = get_bboxes(img.shape, final_rclasses, final_rscores, final_obj)
    removed_boxes = remove_overlapping_objects(objects)
    for i in reversed(removed_boxes):
        final_obj.pop(i)
        final_rclasses.pop(i)
        final_rscores.pop(i)
        final_rnames.pop(i)

    return final_obj, final_rclasses, final_rscores, final_rnames


if __name__ == "__main__":
    from utils.camera.kinect import KinectCamera
    #
    # kinect_right=  KinectCamera(1)
    kinect_left = KinectCamera(0)
    prev_rnames = []

    while True:

        leftimg, left_depth = kinect_left.get_frames()
        #rightimg, right_depth = kinect_right.get_frames()

        dimg = np.rot90(cv2.resize(left_depth, (int(1920 / 2), int(1082 / 2))), 3)
        img = np.rot90(cv2.resize(leftimg, (int(1920 / 2), int(1082 / 2))), 3)


        final_obj, final_rclasses, final_rscores, final_rnames = get_objects(img, dimg)
        objects_taken = set(prev_rnames) ^ set(final_rnames)
        prev_rnames=final_rnames
        print(objects_taken)
        # if 'meixuan_brown_handbag' in final_rnames:
        #     print("mexuian found")
        # else:
        #     print('not found')



        img = img.copy()
        _, _ = visualize_box(img, final_rclasses, final_rscores, final_obj, dimg)
        cv2.imshow("color_img", img)



        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break
