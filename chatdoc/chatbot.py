from typing import Any
import time

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory

from chatdoc.vector_db import VectorDatabase
from chatdoc.citation import Citations, BaseCitation
from chatdoc.embed.embedding_factory import EmbeddingFactory
from chatdoc.chat_model import ChatModel


class Chatbot:
    """
    The chatbot class with a run method
    """

    def __init__(
        self,
        user_id: str,
        embedding_factory: EmbeddingFactory | None = None,
        vector_database: VectorDatabase | None = None,
        memory: ConversationBufferMemory | None = None,
        chat_model: BaseChatModel | None = None,
        last_n_messages: int = 5,
    ) -> None:
        self._embedding_factory = embedding_factory if embedding_factory else EmbeddingFactory()
        self._vector_database = (
            vector_database if vector_database else VectorDatabase(user_id, self._embedding_factory.create())
        )
        self._memory = (
            memory
            if memory
            else ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)
        )
        self.last_n_messages = last_n_messages
        self._chat_model = chat_model if chat_model else ChatModel().chat_model
        self._retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=self._chat_model,
            retriever=self._vector_database.retriever,
            memory=self._memory,
            return_source_documents=True,
        )
        self.embedding_fn = self._embedding_factory.create()
        self.chat_history: list[tuple[str, str, list[BaseCitation]]] = []

    def send_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Method to send a prompt to the chatbot
        """
        result = self._retrieval_chain(
            {
                "question": prompt,
                "chat_history": self.chat_history,
            },
        )
        citations: Citations = Citations(set(), False)
        citations.get_unique_citations(result["source_documents"])
        citations_list = list(citations.citations)
        self.chat_history.append((prompt, result["answer"], citations_list))
        response = {
            "answer": result["answer"],
            "citations": citations_list,
            "chat_history": self.chat_history,
        }
        return response

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
                response = self._retrieval_chain({"question": qry, "chat_history": chat_history})
                print(f"Answer:\n {response['answer']:<{text_width}}", flush=True)
                print("SOURCES: ", flush=True)
                citations: Citations = Citations(set(), with_proof)
                citations.get_unique_citations(response["source_documents"])
                citations.print_citations()
            end = time.time()
            print(f"Time: {end - start:.2f}s", flush=True)
