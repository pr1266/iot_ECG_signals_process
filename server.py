import socketio
from gevent import pywsgi
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import losses
import numpy as np

sio = socketio.Server()
app = socketio.WSGIApp(sio)
model = tf.keras.models.load_model("my_model.hd5f")

@sio.event
def connect(sid, env):
    print(sid, " connected")

@sio.event
def disconnect(sid):
    print(sid, " disconnected")

@sio.on("message")
def get_data(sid, data_value):
    
    model_data = []
    data = data_value['data']
    model_data = np.array(data).reshape(1, 140)
    reconstructions = model(model_data)
    loss = tf.keras.losses.mae(reconstructions, model_data)
    
    preds = tf.math.less(loss, 0.63)
    print("process result : ", preds[0])
    if preds[0] == True:
        device_data = data_value['my_device']
        sio.emit("alert", {'data': device_data})

import os
os.system('cls')
pywsgi.WSGIServer(('', 8080), app).serve_forever()
