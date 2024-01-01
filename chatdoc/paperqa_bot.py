import time
from paperqa import Answer, Docs

from langchain.chat_models.base import BaseChatModel

from .embed.embedding_factory import EmbeddingFactory
from .vector_db import VectorDatabase
from .chat_model import ChatModel


class PaperQABot:
    """
    The chatbot that uses the paperQA library
    """

    def __init__(self, user_id: str) -> None:
        self.embedding_fn = EmbeddingFactory().create()
        self.chat_model: BaseChatModel = ChatModel().chat_model
        self.vector_db = VectorDatabase(user_id, self.embedding_fn)
        self.docs = Docs(llm=self.chat_model, embeddings=self.embedding_fn, doc_index=self.vector_db.chroma_instance)

    def send_prompt(self, prompt: str) -> Answer:
        """
        Method to send a prompt to the chatbot
        """
        result = self.docs.query(prompt)
        return result

    def run(self, text_width: int = 20) -> None:
        """
        Method to run the chatboat using input and printing
        """
        qry = ""
        while qry != "done":
            qry = input("Question: ")
            start = time.time()
            print(f"Initial question:\n {qry:<{text_width}}", flush=True)
            if qry != "done":
                response = self.docs.query(qry)
                print(f"Answer:\n {response.formatted_answer:<{text_width}}", flush=True)
                print("SOURCES: ", flush=True)
                print(response.references, flush=True)
            end = time.time()
            print(f"Time: {end - start:.2f}s", flush=True)
