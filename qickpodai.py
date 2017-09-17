import numpy as np
import cv2
from utils.camera.kinect import KinectCamera


from transitions import Machine as Machine
import pandas as pd

class Qickpod(object):
    states = ['s_partyenter', 's_taken','s_returned', 's_empty']

    transitions = [['t_empty2partyenter', 's_empty', 's_partyenter'],
                   ['t_leave', '*', 's_empty'],
                   ['t_taken2taken', 's_taken', 's_taken'],
                   ['t_returned2returned', 's_returned', 's_returned'],
                   ['t_taken2returned', 's_taken', 's_returned'],
                   ['t_returned2taken', 's_returned', 's_taken']]

    def __init__(self):
        self.machine = Machine(model=self, states=Qickpod.states, transitions=Qickpod.transitions,
                               initial='s_empty', queued=True)
        self.total_party = 0
        self.inventorylist = pd.read_csv('datasheets/productlist.csv')
        self.upcname_dict = pd.Series(demoai.inventorylist.UPC.values, index=demoai.inventorylist.productname).to_dict()
        self.poditems = []

if __name__ == '__main__':
    demoai = Qickpod()

