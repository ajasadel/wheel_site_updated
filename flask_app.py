from flask import Flask, render_template, Response
#from quart import make_response, Quart, render_template, url_for
import cv2
import cv2
import numpy as np
from PIL import Image
import keras.backend.tensorflow_backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
import sys
import threading
import imutils
from keras_retinanet.models.resnet import custom_objects
import configparser
from flask_http2_push import http2push
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
#from yourapplication import app
#import threading
#import json
cwd = os.getcwd()

config = tf.ConfigProto()
#config = tf.ConfigProto(device_count={'GPU': 0})
config.gpu_options.per_process_gpu_memory_fraction = .2
sess = tf.Session(config=config)
class_dictionary = {}
K.clear_session()

alpha=.5
#import tensorflow as tf




# cnt_empty = 0
# cnt_empty1=0
# all_spots=0
#cameras =option_values





app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
CORS(app)
r = redis.Redis(host='localhost', port=6379, db=0)
p = r.pubsub()


def find_camera(id):


    # camerss = json.loads(option_values)
    #print(option_values)
    cameras=['rtsp://admin:Admin@123@192.168.0.146:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif']
    
    return cameras[int(id)]



 

def gen_frames(camera_id):
    #global all_spots,cnt_empty
     
    cam = find_camera(camera_id)
     
    cap=  cv2.VideoCapture(cam)
    camera_id=int(camera_id)
   

    
    while True:
        # for cap in caps:
        # # Capture frame-by-frame
        success, frame = cap.read()
        #imutils.resize(frame,200,200)
        new_image = np.copy(frame)
        overlay = np.copy(frame)


            
            #_,frame1=webcam1.read()
            #cv2.imshow(name,new_image)
            #print("something")
            #cv2.imshow("frame1",frame1)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
               #sys.exit()


         # read the camera frame
        

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg',new_image)
            new_image = buffer.tobytes()
            
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + new_image + b'\r\n') # concat frame one by one and show result
            #yield (cnt_empty)


@app.route('/video_feed/<string:id>/', methods=["GET"])
#@http2push()
def video_feed(id):
   
    """Video streaming route. Put this in the src attribute of an img tag."""
    return  Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feed', methods=["GET"])
#@http2push()
def index():
    return render_template('index.html')
@app.route('/', methods=["GET"])
#@http2push()
def home():
    #global cnt_empty
    return render_template('home.html')
#threading.Thread(target=home()).start()


if __name__ == '__main__':

    app.run(host="127.0.0.1", port=5000)
    socketio.run(app, host='localhost', port=1340,threaded=True,debug=True,ssl_context='adhoc')