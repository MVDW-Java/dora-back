from typing import Any
import time

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory


from .vector_db import VectorDatabase
from .citation import Citations
from .embedding import Embedding
from .chat_model import ChatModel


class Chatbot:
    """
    The chatbot class with a run method
    """

    def __init__(
        self,
        user_id: str,
    ) -> None:
        self.embedding_fn = Embedding().embedding_function
        self.vector_db = VectorDatabase(user_id, self.embedding_fn)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.chat_model: BaseChatModel = ChatModel().chat_model
        self.chatQA = ConversationalRetrievalChain.from_llm(  # pylint: disable=invalid-name
            llm=self.chat_model,
            retriever=self.vector_db.retriever,
            memory=self.memory,
            return_source_documents=True,
        )
        self.chat_history: list[tuple[str, str, Citations]] = []
        self.last_n_messages = 5  # TODO: Change into environment variable

    def send_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Method to send a prompt to the chatbot
        """
        result = self.chatQA(
            {
                "question": prompt,
                "chat_history": self.chat_history[-self.last_n_messages :],
            }
        )
        citations: Citations = Citations(set(), False)
        citations.get_unique_citations(result["source_documents"])
        self.chat_history.append((prompt, result["answer"], citations))
        return result

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
