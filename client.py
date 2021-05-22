import socketio
import time
from prettytable import PrettyTable
import os
sio = socketio.Client()
addr = 'http://localhost:8080'
#! halat e async : 
#! sio = socketio.AsyncClient()

i= 1
@sio.on("alert")
def on_message(data):
    
    t = PrettyTable(['first name', 'last name', 'ward', 'room no.', 'bed no.'])
    data_ = list(data['data'].values())
    t.add_row(data_)
    print("")
    print("***********")
    print("")
    print(f"Alert !")
    print("")
    print("***********")
    print("")
    print(t)

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
    i = 1
    try:
        sio.connect(addr)
        print(f'connected to {addr}, session id : {sio.sid}')
    except:
        print("cannot connect to server, please check the server")
        exit(0)

if __name__ == '__main__':
    main()