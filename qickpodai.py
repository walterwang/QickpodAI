import numpy as np
import time
import sys
import threading
import cv2
from utils.camera.kinect import KinectCamera
from utils.model.ssd_inference import SSD
from threading import Thread
import pandas as pd
from get_object_ssd import get_objects
import requests

from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from transitions.extensions import GraphMachine as Machine
setGlobalLogger(None)
sys.path.append('/home/salil/PycharmProjects/PodClient/')
from camera_manager.queue_manager import QueueManager

class Qickpod(object):
    states = ['s_occupied', 's_taken','s_returned', 's_empty', 's_atdoor']

    transitions = [['t_empty2occupied', 's_empty', 's_occupied'],
                   ['t_empty2atdoor', 's_empty', 's_atdoor'],
                   ['t_atdoor2empty', 's_atdoor', 's_empty'],
                   ['t_atdoor2atdoor', 's_atdoor', 's_atdoor'],
                   ['t_atdoor2occupied', 's_atdoor', 's_occupied'],
                   ['t_occupied2empty', 's_occupied', 's_empty'],
                   ['t_occupied2taken', 's_occupied', 's_taken'],
                   ['t_occupied2returned', 's_occupied', 's_returned'],
                   ['t_returned2occupied', 's_returned', 's_occupied'],
                   ['t_taken2occupied', 's_taken', 's_occupied'],]

    def __init__(self, raw_img=True):
        self.url = 'http://128.122.81.251:8000/post_ai_transaction'
        if raw_img:
            self.raw_kinect_im =True
            self.left_kinect = KinectCamera(0)
            self.right_kinect = KinectCamera(1)

            # rightimg = self.right_kinect.get_frames()[0]
            # rightimg = np.rot90(cv2.resize(rightimg, (rightimg.shape[1]//2, rightimg.shape[0]//2)),1)
            #
            # self.set_template(rightimg, 'door')
            # self.set_template(rightimg, 'pod')
            #
            # leftimg, left_depth = self.left_kinect.get_frames()
            # dimg = np.rot90(cv2.resize(left_depth, (int(1920 / 2), int(1082 / 2))), 3)
            # img = np.rot90(cv2.resize(leftimg, (int(1920 / 2), int(1082 / 2))), 3)
            #
            # self.inventory = self.get_inv(img, dimg)

        else:
            self.raw_kinect_im = False
            self.m = QueueManager.create_manager()
            self.m.connect()
            self.right_kinect = self.m.LatestFrame1()
            self.left_kinect = self.m.LatestFrame0()

            while self.right_kinect.get_latest() is None:
                continue



        rightimg, rightdepth = self.get_rightkinect_img_and_depth()
        self.set_template(rightimg, 'door')
        self.set_template(rightimg, 'pod')

        leftimg, leftdepth = self.get_leftkinect_img_and_depth()
        self.inventory = self.get_inv(leftimg, leftdepth)

        # defining pod variables and camera
        with open("/home/salil/QickpodAI/ssd_path.txt") as f:
            ssd_path = f.readlines()
        self.total_party = 0
        self.headcounter = 0
        self.inventorylist = pd.read_csv(ssd_path[2].strip())
        self.upcname_dict = pd.Series(self.inventorylist.UPC.values, index=self.inventorylist.productname).to_dict()
        self.poditems = []
        # self.ssd_object  = SSD(ssd_path[1].strip(), 21)

        #defining pod fsm:
        self.machine = Machine(model=self, states=Qickpod.states, transitions=Qickpod.transitions,
                               initial='s_occupied', queued=True)

        self.machine.on_enter_s_occupied('onenter_occupied')
        self.machine.on_enter_s_taken('onenter_taken')
        self.machine.on_enter_s_returned('onenter_returned')
        self.machine.on_enter_s_empty('onenter_empty')
        self.machine.on_enter_s_atdoor('onenter_atdoor')

    def get_leftkinect_img_and_depth(self):
        if self.raw_kinect_im:
            leftimg, left_depth = self.left_kinect.get_frames()
        else:
            leftimg_dict = self.left_kinect.get_latest()
            leftimg = leftimg_dict['frame']
            left_depth = leftimg_dict['depth']
        dimg = np.rot90(cv2.resize(left_depth, (int(1920 / 2), int(1082 / 2))), 3)
        img = np.rot90(cv2.resize(leftimg, (int(1920 / 2), int(1082 / 2))), 3)
        return img, dimg

    def get_rightkinect_img_and_depth(self):
        if self.raw_kinect_im:
            rightimg, right_depth = self.right_kinect.get_frames()
        else:
            rightimg_dict = self.right_kinect.get_latest()
            rightimg = rightimg_dict['frame']
            right_depth = rightimg_dict['depth']
        dimg = np.rot90(cv2.resize(right_depth, (int(1920 / 2), int(1082 / 2))), 1)
        img = np.rot90(cv2.resize(rightimg, (int(1920 / 2), int(1082 / 2))), 1)
        return img, dimg

    def consolidate_inv(self, img, depth, sessionid):
        objects_taken = set(self.inventory)^set(self.get_inv(img, depth))
        if self.raw_kinect_im:
            pass
        else:
            for obj in objects_taken:
                data={
                    'session_id': sessionid,
                    'item_name': obj
                }
                print(self.url, data)
                requests.post(self.url, data=data)

        print("object taken", objects_taken)

    def get_inv(self, img, depth):
        final_obj, final_rclasses, final_rscores, final_rnames = get_objects(img, depth)

        return final_rnames
    def set_template(self, img, area):
        if area == 'door':
            self.doortemplate = self.get_area(img, 'door')
        if area == 'pod':
            self.podtemplate = self.get_area(img, 'pod')

    def get_area(self, img, area):
        if area == 'door':
            img_area = img[660:, 0:45]
            img_area =  cv2.GaussianBlur(img_area,(5,5),0)
            return img_area
        if area == 'pod':
            img_area = img[650:, 0:188]
            img_area = cv2.GaussianBlur(img_area, (5, 5), 0)
            return img_area

    def onenter_empty(self):
        while True:
            time.sleep(.5)
            isoccupied, isdoorhead =self.check_heads()
            #print("empty state:", isdoorhead, isoccupied)
            if isdoorhead:
                return self.t_empty2atdoor()
            if isoccupied>0:
                print('head found')
                return self.t_empty2occupied()

    def onenter_occupied(self):
        print('someone entered the store')
        while True:
            if self.check_empty():
                img, dimg = self.get_leftkinect_img_and_depth()

                self.consolidate_inv(img, dimg, self.get_sessionid())
                self.inventory = self.get_inv(img, dimg)
                print("reseting new inventory")

                return self.t_occupied2empty()
    def get_sessionid(self):
        if self.raw_kinect_im:
            return 0
        else:
            leftimg_dict = self.left_kinect.get_latest()
            return leftimg_dict['session_id']


    def onenter_atdoor(self):
        isoccupied, isatdoor = self.check_heads()
        if isatdoor:
            time.sleep(.3)
            return self.t_atdoor2atdoor()
        elif isoccupied:
            print('from atdoor to occupied, ')
            return self.t_atdoor2occupied()
        else:
            return self.t_atdoor2empty()
        print('error returning to onenterdoor')
        return self.t_atdoor2atdoor()

    def get_depth_area(self, depth_matrix):


        try:
            # cv2.imshow('depth', depth_matrix)
            # cv2.waitKey(1)
            depth_matrix[depth_matrix == np.inf] = np.nan
            depth = int(np.nanmean(
                depth_matrix[594:, 0:188]))
        except ValueError:

            depth = 0

        return depth

    def check_heads(self):

        rightimg, rightdepth = self.get_rightkinect_img_and_depth()
        podimg = self.get_area(rightimg, 'pod')
        doorimg = self.get_area(rightimg, 'door')
        depthofpod = self.get_depth_area(rightdepth)
        print("depth of pod", depthofpod)

        isoccupied, isatdoor = False, False

        #cv2.imshow('pod', podimg)
        #cv2.imshow('doorimg', doorimg)
        #cv2.waitKey(1)
        # check to see if doorimg match with doorimg.template()
        res = cv2.matchTemplate(podimg, self.podtemplate, cv2.TM_CCOEFF_NORMED)
        _, matchscore, _, _ = cv2.minMaxLoc(res)

        if matchscore < .83:
            isoccupied = True
        print('pod', matchscore)
        res = cv2.matchTemplate(doorimg, self.doortemplate, cv2.TM_CCOEFF_NORMED)
        _, matchscore, _, _ = cv2.minMaxLoc(res)
        if matchscore < .76 and depthofpod>1900:
            isatdoor = True

        print('res dooronly', matchscore)
        return isoccupied, isatdoor

    def check_empty(self):
        for i in range(3):
            time.sleep(.5)
            if self.check_heads()[0]:
                return False
        return True


if __name__ == '__main__':
    demoai = Qickpod()

    def show_graph():
        while True:
            graph_pic = demoai.get_graph().draw(format='png', prog='dot')
            g = np.fromstring(graph_pic, np.uint8)
            img = cv2.imdecode(g, cv2.IMREAD_COLOR)


            cv2.imshow("dynamicgraph", img)
            key = cv2.waitKey(delay=1)
            if key == ord('q'):
                break

    t = threading.Thread(target=demoai.t_occupied2empty)
    m = threading.Thread(target=show_graph)

    t.start()
    m.start()

    sys.exit(0)

