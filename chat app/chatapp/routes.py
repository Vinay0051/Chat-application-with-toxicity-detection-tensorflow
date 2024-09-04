from flask import Blueprint, render_template
from flask import Flask, request, jsonify
import json

main = Blueprint("main", __name__)

@main.route("/")
def base():
    return render_template("base.html")

@main.route("/desktop-1.html")
def desktop_1():
    return render_template("desktop-1.html")

@main.route("/index.html")
def index():
    return render_template("index.html")

@main.route("/login.html", methods=['GET','POST'])
def login():
    return render_template("login.html")

@main.route("/signup.html")
def signup():
    return render_template("signup.html")

@main.route("/welcome.html")
def welcome():
    return render_template("welcome.html")

@main.route("/result.html")
def result():
    return render_template("result.html")

@main.route("/validate_credentials", methods=["POST"])
def validate_credentials():
    # Load user data from JSON file
    with open(r"C:\Users\vicky\Desktop\SDP\chat app\chatapp\users.json", "r") as file:
        users_data = json.load(file)

    username = request.form.get("username")
    password = request.form.get("password")

    # Check if username and password match any user in the JSON file
    for user in users_data["users"]:
        if user["username"] == username and user["password"] == password:
            return jsonify({"status": "success", "message": "Credentials are valid"})

    return jsonify({"status": "error", "message": "Invalid credentials"})

@main.route('/store_credentials', methods=['POST'])
def store_credentials():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Load existing users data from the users.json file
        with open(r'C:\Users\vicky\Desktop\SDP\chat app\chatapp\users.json', 'r') as file:
            users = json.load(file)

        # Check if the username already exists
        for user in users['users']:
            if user['username'] == data['username']:
                return jsonify({'status': 'error', 'message': 'Username already exists'})

        # Add the new user to the users list
        new_user = {"username": data['username'], "password": data['password']}
        users['users'].append(new_user)

        # Write the updated users list back to the users.json file
        with open(r'C:\Users\vicky\Desktop\SDP\chat app\chatapp\users.json', 'w') as file:
            json.dump(users, file, indent=4)

        return jsonify({'status': 'success', 'message': 'Signup successful'})
    except Exception as e:
        print(str(e))
        return jsonify({'status': 'error', 'message': 'Error occurred'})