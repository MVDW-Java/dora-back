from flask import Flask, request, session
import tempfile
from pathlib import Path
import uuid
from datetime import date
from chatdoc.document_loader import DocumentLoader
from chatdoc.vector_db import VectorDatabase
from chatdoc.embedding import Embedding

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

@app.route('/identify', methods=['POST'])
def identify():
    response = {
            "sessionId": "",
            "message": "",
            "hasDB": False,
        }
    if 'id' in session and isinstance(session['id'], str):
        session_id: str = session['id']
        response['sessionId'] = session_id
        response['message'] = 'Welcome back user: ' + session_id + '!'
    else:
        response['sessionId'] = str(uuid.uuid4())
        response['message'] = 'Welcome new user: ' + response['sessionId'] + '!'
    return response


async def process_files(document_dict: dict[str, Path], user_id: str) -> None:
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
    files: FileStorage = request.files['files']
    dir_path: Path = Path(tempfile.gettempdir()) / Path(str(session['id']))
    full_document_dict = {}
    for file in files:
        unique_file_name = f"{file.stem}_{current_date}_{file.suffix}"
        unique_file_path = dir_path / Path(unique_file_name)
        full_document_dict[file.name] = unique_file_path
        file.save()
    
    if len(files) == 1:
        return 'File uploaded successfully!'
    return str(len(files)) + ' files uploaded successfully!'
    

@app.route('/message', methods=['POST'])
def save_message():
    """
    Save the received message to a file.

    Returns:
        str: A success message if the message is saved successfully.
        str: An error message if no JSON body is received.
    """
    if request.json is not None:
        message = request.json['message']
        with open('llm.txt', 'a') as file:
            file.write(message + '\n')
        return 'Message saved successfully!'
    return 'No JSON body received', 400

if __name__ == '__main__':
    app.run()
