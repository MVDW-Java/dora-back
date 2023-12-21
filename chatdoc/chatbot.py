from typing import Any, cast
from pathlib import Path
import os
import time

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.embeddings.openai import OpenAIEmbeddings


from paperqa import Docs # type: ignore

from .document_loader import DocumentLoader
from .vector_db import VectorDatabase
from .citation import Citations
from .utils import Utils




class Chatbot:
    """
    The chatbot class with a run method
    """

     _chroma_client = chromadb.PersistentClient()

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        """
        ChromaDB client
        """
        return self._chroma_client

    def __init__(
        self,
        document_dict: dict[str, Path],
        chat_model_name: str = "gpt-3.5-turbo",
        embedding_model_name: str = "text-embedding-ada-002",
        **kwargs: dict[str, Any],
    ):
        embeddings = OpenAIEmbeddings(api_key=Utils.read_api_key(kwargs), model=embedding_model_name)
        current_collection = self.chroma_client.get_or_create_collection(collection_name)
        document_loader = DocumentLoader(model_name=embedding_model_name, file_paths=document_dict.values())
        current_collection.add_documents(document_loader.split_data, embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.chat_model: BaseChatModel = cast(
            BaseChatModel,
            kwargs.get(
                "custom_chat_model",
                ChatOpenAI(
                    api_key=Utils.read_api_key(kwargs),
                    temperature=0,
                    model=cast(str, kwargs.get("chat_model_name", chat_model_name)),
                ),
            ),
        )
        self.chatQA = ConversationalRetrievalChain.from_llm(  # pylint: disable=invalid-name
            llm=self.chat_model,
            retriever=vector_db.vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

    def send_prompt(self, prompt: str) -> str:
        """
        Method to send a prompt to the chatbot
        """
        return self.chat_model.send_prompt(prompt)

    def run(self, text_width: int = 20, with_proof: bool = False):
        """
        Method to run the chatboat using input and printing
        """
        chat_history = []
        qry = ""
        while qry != "done":
            qry = input("Question: ")
            start = time.time()
            print(f"Initial question:\n {qry:<{text_width}}", flush=True)
            if qry != "done":
                response = self.chatQA({"question": qry, "chat_history": chat_history})
                print(f"Answer:\n {response['answer']:<{text_width}}", flush=True)
                print("SOURCES: ", flush=True)
                citations: Citations = Citations(set(), with_proof)
                citations.get_unique_citations(response["source_documents"])
                citations.print_citations()
            end = time.time()
            print(f"Time: {end - start:.2f}s", flush=True)
   