import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from shutil import copyfile
import shutil
import os, glob
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from flask import Flask, render_template, Response

framework="tflite"  #tf, tflite, trt
model="yolov4"  #yolov3 or yolov4
tiny=True      #yolo or yolo-tiny
iou=0.45        #iou threshold
score=0.25      #score threshold
output='./detections/'  #path to output folder

#def main():
app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 416
    weights_loaded="./checkpoints/test.tflite"
    print("Load video từ webcam" )
    vid = cv2.VideoCapture(1)

    if framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=weights_loaded)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    
    
    # if output:
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(vid.get(cv2.CAP_PROP_FPS))
    #     codec = cv2.VideoWriter_fourcc(*output_format)
    #     out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_id = 0
    while (vid.isOpened()):
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)

        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")


        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,input_shape=tf.constant([input_size, input_size]))                                                                  
            print('Fire')    
                           
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        print("FPS: {0}".format(fps))
        
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("result", result)
        cv2.imwrite('demo.jpg', result)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'): break
# def result():
#     result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
#     cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("result", result)
#         # if cv2.waitKey(1) & 0xFF == ord('q'): break
#         # if output:
#         #     out.write(result)
#     frame_id += 1
           
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
                    


if __name__ == '__main__':
    app.run(debug=True)