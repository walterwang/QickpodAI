import numpy as np
import time
import sys
import threading
import cv2
from utils.camera.kinect import KinectCamera
from utils.model.ssd_inference import SSD
import pandas as pd

from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from transitions.extensions import GraphMachine as Machine
setGlobalLogger(None)
#from camera_manager.queue_manager import QueueManager

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
        if raw_img:
            self.raw_kinect_im =True
            self.left_kinect = KinectCamera(1)
            self.right_kinect = KinectCamera(0)

            rightimg = self.right_kinect.get_frames()[0]
            rightimg = np.rot90(cv2.resize(rightimg, (rightimg.shape[1]//2, rightimg.shape[0]//2)),1)
            self.set_template(rightimg, 'door')
            self.set_template(rightimg, 'pod')


        else:
            print("Zeleng's queue manager not implemented")
            sys.exit(0)
            # self.m = QueueManager.create_manager()
            # self.m.connect()
            # self.right_kinect = self.m.LatestFrame1()
            # self.left_kinect = self.m.LatestFrame0()
            pass

        # defining pod variables and camera
        with open("ssd_path.txt") as f:
            ssd_path = f.readlines()
        self.total_party = 0
        self.headcounter = 0
        self.inventorylist = pd.read_csv(ssd_path[2].strip())
        self.upcname_dict = pd.Series(self.inventorylist.UPC.values, index=self.inventorylist.productname).to_dict()
        self.poditems = []
        self.ssd_object  = SSD(ssd_path[1].strip(), 21)

        #defining pod fsm:
        self.machine = Machine(model=self, states=Qickpod.states, transitions=Qickpod.transitions,
                               initial='s_occupied', queued=True)

        self.machine.on_enter_s_occupied('onenter_occupied')
        self.machine.on_enter_s_taken('onenter_taken')
        self.machine.on_enter_s_returned('onenter_returned')
        self.machine.on_enter_s_empty('onenter_empty')
        self.machine.on_enter_s_atdoor('onenter_atdoor')

    def construct_inventory(self):

        pass
    def set_template(self, img, area):
        if area == 'door':
            self.doortemplate = self.get_area(img, 'door')
        if area == 'pod':
            self.podtemplate = self.get_area(img, 'pod')

    def get_area(self, img, area):
        if area == 'door':
            img_area = img[635:, 0:30]
            return img_area
        if area == 'pod':
            img_area = img[635:, 0:166]
            return img_area

    def onenter_empty(self):
        while True:
            time.sleep(.5)
            isoccupied, isdoorhead =self.check_heads()
            print("empty state:", isdoorhead, isoccupied)
            if isdoorhead:
                return self.t_empty2atdoor()
            if isoccupied>0:
                print('head found')
                return self.t_empty2occupied()

    def onenter_occupied(self):
        print('someone entered the store')
        while True:
            if self.check_empty():
                return self.t_occupied2empty()

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

    def check_heads(self):

        if self.raw_kinect_im:
            rightimg = self.right_kinect.get_frames()[0]
            rightimg = np.rot90(cv2.resize(rightimg, (rightimg.shape[1]//2, rightimg.shape[0]//2)),1)
            podimg = self.get_area(rightimg, 'pod')
            doorimg = self.get_area(rightimg, 'door')

        else:
            # use Zeleng's half sized images
            pass
        isoccupied, isatdoor = False, False

        # check to see if doorimg match with doorimg.template()
        res = cv2.matchTemplate(podimg, self.podtemplate, cv2.TM_CCOEFF_NORMED)
        _, matchscore, _, _ = cv2.minMaxLoc(res)
        if matchscore < .8:
            isoccupied = True
        print('maxval', matchscore)
        res = cv2.matchTemplate(doorimg, self.doortemplate, cv2.TM_CCOEFF_NORMED)
        _, matchscore, _, _ = cv2.minMaxLoc(res)
        if matchscore < .8:
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

