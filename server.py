from flask import Flask, request
import os
import tempfile
from pathlib import Path
import uuid
from flask import session
from datetime import date

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

@app.route('/identify', methods=['POST'])
def identify():
    response = {
            "sessionId": "",
            "message": ""
        }
    response
    if 'id' in session:
        response['sessionId'] = session['id']
        response['message'] = 'Welcome back user: ' + session['id'] + '!'
    else:
        response['sessionId'] = str(uuid.uuid4())
        response['hasDB'] = False
        response['message'] = 'Welcome new user: ' + response['sessionId'] + '!'
    return response


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'id' not in session:
        return 401, "You have not been authenticated, please identify yourself first."
    current_date: string = date.today().strftime('%Y-%m-%d')
    files = request.files['files']
    dir_path = Path(tempfile.gettempdir()) / session['id']
    full_document_dict = {}
    for file in files:
        unique_file_name = file.stem / f"_{current_date}_" / file.suffix
        unique_file_path = dir_path / Path(unique_file_name)
        full_document_dict[file.name] = unique_file_path
        file.save()
    if len(files) == 1:
        return 'File uploaded successfully!'
    return str(len(files)) + ' files uploaded successfully!'
    

@app.route('/message', methods=['POST'])
def save_message():
    message = request.json['message']
    with open('llm.txt', 'a') as file:
        file.write(message + '\n')
    return 'Message saved successfully!'

if __name__ == '__main__':
    app.run()
