from flask import Flask
from threading import Thread
app = Flask(__name__)
from qickpodai import Qickpod
import cv2
import numpy as np
import time
port = 14555

status = 1


@app.route('/walterAI/podstate')
def podstate():
    return demoai.state

@app.route('/walterAI/turnon')
def turnon():
    return 'success'

@app.route('/walterAI/turnoff')
def turnoff():
    return 'success'


def show_graph():
    while True:
        graph_pic = demoai.get_graph().draw(format='png', prog='dot')
        g = np.fromstring(graph_pic, np.uint8)
        img = cv2.imdecode(g, cv2.IMREAD_COLOR)
        cv2.imshow("dynamicgraph", img)
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break

if __name__ == '__main__':
    demoai = Qickpod()
    Thread(target=demoai.t_occupied2empty).start()
    Thread(target=show_graph).start()
    Thread(app.run(port=port).start())

