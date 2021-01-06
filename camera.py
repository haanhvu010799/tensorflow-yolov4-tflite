################################################################################

# Example : perform live fire detection in video using FireNet CNN

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import threading
import requests
import json

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################
global ThingsBoardFrame
class VideoCamera(object):    
    def __init__(self):
        global video
        global model
        global rows
        global cols
        global width
        global height
        global fps
        global frame_time
        global onFire
        global ThingsBoardFrame
        global ThingsBoardHost
        global headers
        ThingsBoardHost = 'http://localhost:8080/api/v1/iayGzRp8hhzJzTPWGGcy/telemetry'
        headers = {'Content-type': 'application/json'}
        onFire = False
        # construct and display model
        def construct_firenet (x,y, training=False):
        # Build network as per architecture in [Dunnings/Breckon, 2018]

            network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

            network = conv_2d(network, 64, 5, strides=4, activation='relu')

            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)

            network = conv_2d(network, 128, 4, activation='relu')

            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)

            network = conv_2d(network, 256, 1, activation='relu')

            network = max_pool_2d(network, 3, strides=2)
            network = local_response_normalization(network)

            network = fully_connected(network, 4096, activation='tanh')
            if(training):
                network = dropout(network, 0.5)

            network = fully_connected(network, 4096, activation='tanh')
            if(training):
                network = dropout(network, 0.5)

            network = fully_connected(network, 2, activation='softmax')

            # if training then add training hyperparameters

            if(training):
                network = regression(network, optimizer='momentum',
                                    loss='categorical_crossentropy',
                                    learning_rate=0.001)

            # constuct final model

            model = tflearn.DNN(network, checkpoint_path='firenet',
                                max_checkpoints=1, tensorboard_verbose=2)

            return model
        
        model = construct_firenet (224, 224, training=False)
        print("Constructed FireNet ...")

        model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
        print("Loaded CNN network weights ...")
        
        rows = 224
        cols = 224
        
        video = cv2.VideoCapture(0)

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps);
        
        

    ################################################################################

    def detectFire(self):
        print("Start detecting fire")
        while True:
            # display and loop settings
            
            ret, frame = video.read()
            
            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

            output = model.predict([small_frame])

            if round(output[0][0]) == 1:
                    cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                    cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
                    self.onFire = True
                    #Gui alarm bao chay den thingsboard
                    alarmJson = {"onFire": True}
                    r = requests.post(ThingsBoardHost, data=json.dumps(alarmJson), headers=headers)
                    print("ON FIRE! Send alarm to server HTTP code: " + str(r.status_code))
            else:
                    cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                    cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
                    self.onFire = False
                    alarmJson = {"onFire": False}
                    r = requests.post(ThingsBoardHost, data=json.dumps(alarmJson), headers=headers)

            ret, jpeg = cv2.imencode('.jpg', frame)
            self.ThingsBoardFrame = jpeg.tobytes()
    
    ################################################################################
            
    def startDetectFire(self):
        threading.Thread(target=self.detectFire).start()
    
    def getOnFire(self):
        return self.onFire
    
    ################################################################################
        
    def getThingsBoardFrame(self):
        return self.ThingsBoardFrame

################################################################################

from flask import Flask, Response
import threading

global camera
camera = VideoCamera()

fireDetectionThread = threading.Thread(target=camera.detectFire)

def create_app():
    app = Flask(__name__)
    
    @app.route("/")
    def index():
        return "Hello world!"

    def gen():
        while True:
            frame = camera.getThingsBoardFrame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
    @app.route("/video_feed")
    def video_feed():
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def checkFire():
        global fireDetectionThread
        fireDetectionThread.setDaemon(True)
        fireDetectionThread.start()
    
    checkFire()
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
