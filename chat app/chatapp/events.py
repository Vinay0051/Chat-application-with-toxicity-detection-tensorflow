from flask import request
from flask_socketio import emit
from .extensions import socketio
from model_re import main

users = {}

@socketio.on("connect")
def handle_connect():
    print("Client connected!")

@socketio.on("user_join")
def handle_user_join(username):
    print(f"User {username} joined!")
    users[username] = request.sid

@socketio.on("new_message")
def handle_new_message(message):
    print(f"New message: {message}")
    username = None 
    for user in users:
        if users[user] == request.sid:
            username = user
    msg=f"{username} : {message}"
    print(msg)
    file_object=open(r"C:\Users\vicky\Desktop\SDP\chat app\chatapp\conversation.txt","a+")
    file_object.write(msg)
    file_object.write('\n')
    file_object.close()
    emit("chat", {"message": message, "username": username}, broadcast=True)

@socketio.on('run_python_script')
def handle_run_python_script():
    result=main()