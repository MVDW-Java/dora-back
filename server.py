# system imports
import tempfile
from pathlib import Path
import uuid
import os
import asyncio
from datetime import date
import logging

# third party imports
from flask import Flask, request, session, make_response, Response
from werkzeug.utils import secure_filename
from flask_cors import CORS

# local imports
from chatdoc.doc_loader.document_loader import DocumentLoader
from chatdoc.doc_loader.document_loader_factory import DocumentLoaderFactory
from chatdoc.vector_db import VectorDatabase
from chatdoc.embed.embedding_factory import EmbeddingFactory
from chatdoc.chatbot import Chatbot

app = Flask(__name__)
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True

current_env = os.environ.get("CURRENT_ENV", "DEV")

list_of_allowed_origins: list[str] = []
app.logger.level = logging.INFO
match current_env:
    case "DEV":
        CORS(app)
        if os.environ.get("DEV_URL") is not None:
            list_of_allowed_origins.append(os.environ["DEV_URL"])
        else:
            raise ValueError("DEV_URL environment variable not set")
        app.logger.log(level=logging.INFO, msg="Running in development mode")
    case "PROD":
        app.logger.log(level=logging.INFO, msg="Running in production mode")
    case _:
        raise ValueError("Invalid environment variable set for CURRENT_ENV")

app.secret_key = str(uuid.uuid4())


async def process_files(document_dict: dict[str, Path], user_id: str) -> None:
    """
    Process the files in the given document dictionary and add them to the vector database.

    Args:
        document_dict (dict[str, Path]): A dictionary mapping document names to their file paths.
        user_id (str): The ID of the user.

    Returns:
        None
    """
    loader_factory = DocumentLoaderFactory()
    document_loader = DocumentLoader(document_dict, loader_factory)
    embedding_fn = EmbeddingFactory().create()
    vector_db = VectorDatabase(user_id, embedding_fn)
    documents = document_loader.text_splitter.split_documents(document_loader.document_iterator)
    app.logger.log(level=logging.INFO, msg="Adding documents to vector database...")
    await vector_db.add_documents(documents)


@app.errorhandler(ValueError)  # type: ignore
def handle_value_error(error: ValueError) -> Response:
    """
    Handles a ValueError.

    Args:
        error (ValueError): The ValueError to handle.

    Returns:
        Response: A response object containing the error message and status code.
    """
    return make_response({"error": str(error)}, 400)


@app.after_request
def add_cors_headers(response: Response) -> Response:
    """
    Adds CORS headers to the response object.

    Args:
        response (Response): The response object to add CORS headers to.

    Returns:
        Response: The response object with CORS headers added.
    """
    origin = request.headers.get("Origin")
    if origin in list_of_allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.route("/identify", methods=["GET"])
def identify() -> Response:
    """
    Identifies the user and returns a response object.

    Returns:
        dict: A response object containing the sessionId and message.
    """
    response = {}
    if "id" in session and isinstance(session["id"], str):
        response["message"] = "Welcome back user: " + session["id"] + "!"
        response["sessionId"] = session["id"]
        response["authenticated"] = True
        response["hasDB"] = session["hasDB"]
    else:
        session["id"] = str(uuid.uuid4())
        session["authenticated"] = True
        session["hasDB"] = False
        response["message"] = "Welcome new user: " + session["id"] + "!"
    response = make_response(response)
    return response


@app.route("/upload_files", methods=["OPTIONS"])
def upload_files_options() -> Response:
    """
    Handles the OPTIONS request for the upload_files route.

    Returns:
        Response: A response object containing the CORS headers.
    """
    response = make_response("OPTIONS OK", 200)
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    return response


@app.route("/upload_files", methods=["POST"])
def upload_files() -> Response:
    """
    Uploads files to the server.

    Returns:
        str: Success message indicating the number of files uploaded.
        tuple: Error message and status code if user is not authenticated.
    """

    if "id" not in session:
        return make_response({"error": "You have not been authenticated, please identify yourself first."}, 401)
    current_date: str = date.today().strftime("%Y-%m-%d")
    prefix: str = request.form["prefix"]
    files = {k.lstrip(prefix): v for k, v in request.files.items() if k.startswith(prefix)}
    dir_path: Path = Path(tempfile.gettempdir()) / Path(str(session["id"]))
    os.makedirs(dir_path, exist_ok=True)
    full_document_dict = {}
    for filename, file in files.items():
        secure_file_name = secure_filename(filename)
        secure_path = Path(secure_file_name)
        unique_file_name = f"{secure_path.stem}_{current_date}_{secure_path.suffix}"
        unique_file_path = dir_path / Path(unique_file_name)
        full_document_dict[unique_file_name] = unique_file_path
        file.save(unique_file_path)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(process_files(full_document_dict, session["id"]))
    response = make_response(
        {"message": f"{str(len(files))} file{'s' if len(files) != 1 else ''} uploaded successfully!"}, 200
    )
    return response


@app.route("/prompt", methods=["OPTIONS"])
def prompt_options() -> Response:
    """
    Handles the OPTIONS request for the prompt route.

    Returns:
        Response: A response object containing the CORS headers.
    """
    response = make_response()
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    return response


@app.route("/prompt", methods=["POST"])
def prompt() -> Response:
    """
    This function handles the prompt request from the client.

    Returns:
        tuple: A tuple containing the response message and the HTTP status code.
    """
    if "id" not in session:
        return make_response({"error": "You have not been authenticated, please identify yourself first."}, 401)
    if request.json is None:
        return make_response({"error": "No JSON body received"}, 400)
    message = request.json["prompt"]
    chatbot = Chatbot(session["id"])
    result = chatbot.send_prompt(message)
    return make_response(result, 200)


if __name__ == "__main__":
    app.run()
