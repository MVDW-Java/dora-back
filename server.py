import tempfile
from pathlib import Path
import uuid
import os
from datetime import date
from flask import Flask, request, session
from werkzeug.utils import secure_filename
from chatdoc.document_loader import DocumentLoader
from chatdoc.vector_db import VectorDatabase
from chatdoc.embedding import Embedding
from chatdoc.chatbot import Chatbot

current_env = os.environ.get('CURRENT_ENV', 'DEV')



app = Flask(__name__)

match current_env:
    case 'DEV':
        from flask_cors import CORS
        CORS(app)
        print('Running in development mode')
    case 'PROD':
        print('Running in production mode')
    case _:
        raise ValueError('Invalid environment variable set for CURRENT_ENV')

app.secret_key = str(uuid.uuid4())

@app.route('/', methods=['OPTIONS'])
def options_backdoor():
    """
    Backdoor for testing CORS.
    """
    request.headers.add('Access-Control-Allow-Origin', '*')
    return 'OK'

@app.route('/identify', methods=['GET'])
def identify():
    """
    Identifies the user and returns a response object.

    Returns:
        dict: A response object containing the sessionId and message.
    """
    response = {}
    if 'id' in session and isinstance(session['id'], str):
        response['message'] = 'Welcome back user: ' + session['id'] + '!'
        response['sessionId'] = session['id']
        response['authenticated'] = True
        response['hasDB'] = session['hasDB']
    else:
        session['id'] = str(uuid.uuid4())
        session['authenticated'] = True
        session['hasDB'] = False
        response['message'] = 'Welcome new user: ' + session['id'] + '!'
    return response


async def process_files(document_dict: dict[str, Path], user_id: str) -> None:
    """
    Process the files in the given document dictionary and add them to the vector database.

    Args:
        document_dict (dict[str, Path]): A dictionary mapping document names to their file paths.
        user_id (str): The ID of the user.

    Returns:
        None
    """
    document_loader = DocumentLoader(document_dict)
    documents = document_loader.text_splitter.split_documents(document_loader.document_iterator)
    embedding_fn = Embedding().embedding_function
    vector_db = VectorDatabase(user_id, embedding_fn)
    await vector_db.add_documents(documents)



@app.route('/upload_files', methods=['POST'])
def upload_files():
    """
    Uploads files to the server.

    Returns:
        str: Success message indicating the number of files uploaded.
        tuple: Error message and status code if user is not authenticated.
    """
    
    if 'id' not in session:
        return "You have not been authenticated, please identify yourself first.", 401
            
    current_date: str = date.today().strftime('%Y-%m-%d')
    prefix: str = request.form['prefix']
    files = {k.lstrip(prefix):v for k,v in request.files.items() if k.startswith(prefix)}
    dir_path: Path = Path(tempfile.gettempdir()) / Path(str(session['id']))
    full_document_dict = {}
    for filename, file in files.items():
        secure_file_name = secure_filename(filename)
        secure_path = Path(secure_file_name)
        unique_file_name = f"{secure_path.stem}_{current_date}_{secure_path.suffix}"
        unique_file_path = dir_path / Path(unique_file_name)
        full_document_dict[file.name] = unique_file_path
        file.save(unique_file_path)
    if len(files) == 1:
        return 'File uploaded successfully!'
    return str(len(files)) + ' files uploaded successfully!'
   

@app.route('/prompt', methods=['POST'])
def prompt():
    """
    This function handles the prompt request from the client.

    Returns:
        tuple: A tuple containing the response message and the HTTP status code.
    """
    if 'id' not in session:
        return "You have not been authenticated, please identify yourself first.", 401
    if request.json is None:
        return 'No JSON body received', 400
    message = request.json['prompt']
    chatbot = Chatbot(session['id'])
    response = chatbot.send_prompt(message)
    return response

if __name__ == '__main__':
    app.run()
