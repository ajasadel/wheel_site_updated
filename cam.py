import cv2
from flask_cors import CORS
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from run_anpr_demo6 import ANPRConcor
from config import camera
import json
import ast
from threading import Thread
from multiprocessing import Process
import pdb
import redis
import time


# pdb.set_trace()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
CORS(app)
r = redis.Redis(host='localhost', port=6379, db=0)
p = r.pubsub()
#p.subscribe("msg")



def run_wheel(r):
    #ac = ANPRConcor()

    cam_ip1 = 'rtsp://admin:Admin@123@192.168.0.146:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
    cam1 = cv2.VideoCapture(cam_ip1)
    while true:
        cam


# f = open('roi.txt', 'w')
# f.write(json.dumps(roi_dict))
# f.close()


@app.route("/")
def index():
        return render_template("index")


@socketio.on("msg")
def handle_message(message):
    socketio.emit("data", {"msg": message})


#         socketio.emit("data1", command)
def my_handler(msg):
    msg_dict = ast.literal_eval(msg['data'].decode())
    socketio.emit("data1", msg_dict)


p.subscribe(**{'msg': my_handler})
thread = p.run_in_thread(sleep_time=0.001)


if __name__ == '__main__':
    p = Process(target=run_wheel, args=(r,))
    p.start()
    socketio.run(app, host='localhost', port=1340,threaded=True,debug=True,ssl_context='adhoc')