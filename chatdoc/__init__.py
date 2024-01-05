import asyncio
from pathlib import Path
from tempfile import gettempdir
from chatdoc.embed.embedding_factory import EmbeddingFactory
from werkzeug.utils import secure_filename
from chatdoc.doc_loader.document_loader import DocumentLoader  # TODO: remove these lines before commit
from chatdoc.doc_loader.document_loader_factory import DocumentLoaderFactory  # TODO: remove these lines before commit
from chatdoc.chatbot import Chatbot
from chatdoc.vector_db import VectorDatabase  # TODO: remove these lines before commit


def main():
    user_identification = "adr_collection"  # TODO: remove this line
    temp_dir = gettempdir()
    raw_names = ["./Interim-auditrapport 2022 Financien.pdf", "./Interim-auditrapport 2023 Financien.pdf"]
    safe_names = [secure_filename(raw_name) for raw_name in raw_names]
    document_dict = {
        safe_names[0]: Path(raw_names[0]),
        safe_names[1]: Path(raw_names[1]),
    }
    document_dict_after = {k: str(v) for k, v in document_dict.items()}
    embedding_fn = EmbeddingFactory().create()
    vector_db = VectorDatabase(user_identification, embedding_fn)

    # Store files in the Vector DB
    # document_loader = DocumentLoader(document_dict, DocumentLoaderFactory())
    # documents = document_loader.text_splitter.split_documents(document_loader.document_iterator)
    # loop = asyncio.new_event_loop()
    # loop.run_until_complete(vector_db.add_documents(documents))
    # loop.close()

    # Create chatbot
    chat_bot = Chatbot(user_identification, document_dict_after)
    chat_bot.send_prompt("What is the difference between the provided documents?")


if __name__ == "__main__":
    main()
