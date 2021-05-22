import socketio
from csv import reader
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.eager.context import device

model = tf.keras.models.load_model("my_model.hd5f")

sio = socketio.Client()
device_data = {
    "first_name" : "pourya",
    "last_name" : "pooryeganeh",
    "ward" : "centeral",
    "room" : "1266",
    "bed" : "4",
}

@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")

def main():

    sio.connect('http://localhost:8080')
    print('my sid is ', sio.sid)
    dataframe = pd.read_csv('ecg.csv', header = None)
    raw_data = dataframe.values
    labels = raw_data[:,-1]
    data = raw_data[:, 0:-1]
    
    train_data, test_data, train_labels, test_labels = train_test_split(
        data,
        labels,
        test_size = 0.2,
        random_state = 21
    )
    
    for i in range(len(test_data)):
        print("")
        print("***********")
        print("ECG information of : ")
        print(f"{device_data['first_name']} {device_data['last_name']}")
        print(f"ward : {device_data['ward']}, room : {device_data['room']}, bed : {device_data['bed']}")
        print(f"time : ", time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
        print("***********")
        print("")
        sio.emit('message', {'data': list(test_data[i]), 'my_device' : device_data})
        time.sleep(1)

if __name__ == '__main__':
    main()