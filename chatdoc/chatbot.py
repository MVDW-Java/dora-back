from typing import Any
from logging import Logger
import time
from pathlib import Path

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field

from chatdoc.vector_db import VectorDatabase
from chatdoc.citation import Citations, BaseCitation, Citation
from chatdoc.embed.embedding_factory import EmbeddingFactory
from chatdoc.chat_model import ChatModel


class DocumentInput(BaseModel):
    question: str = Field()


class Chatbot:
    """
    The chatbot class with a run method
    """

    def __init__(
        self,
        user_id: str,
        document_dict: dict[str, str],
        embedding_factory: EmbeddingFactory | None = None,
        vector_database: VectorDatabase | None = None,
        memory: ConversationBufferMemory | None = None,
        chat_model: BaseChatModel | None = None,
        last_n_messages: int = 5,
        logger: Logger | None = None,
    ) -> None:
        self._embedding_factory = embedding_factory if embedding_factory else EmbeddingFactory()
        self._embedding_fn = self._embedding_factory.create()
        self._vector_database = vector_database if vector_database else VectorDatabase(user_id, self._embedding_fn)
        self.chain_memory = (
            memory
            if memory
            else ConversationBufferMemory(memory_key="chain_history", output_key="answer", return_messages=True)
        )
        self.agent_memory = ConversationBufferMemory(
            memory_key="agent_history", output_key="output", return_messages=True
        )
        self.last_n_messages = last_n_messages
        self._chat_model = chat_model if chat_model else ChatModel().chat_model
        self.tools = self.create_tools(document_dict)
        self.agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=self.tools,
            llm=self._chat_model,
            memory=self.agent_memory,
            # verbose=True,
            return_intermediate_steps=True,
            return_source_documents=True,
        )
        self.logger = logger if logger else Logger("chatbot")
        self.chat_history_internal: list = []
        self.chat_history_export: list[tuple[str, str, list[BaseCitation]]] = []

    def create_tools(self, document_dict: dict[str, str]) -> list[Tool]:
        document_names = [str(Path(document_name).stem) for document_name in document_dict.keys()]
        retrieval_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=self._chat_model,
            retriever=self._vector_database.retriever,
            memory=self.chain_memory,
            return_source_documents=True,
            # verbose=True,
        )

        def run_chain(question: str) -> dict[str, Any]:
            results = retrieval_chain({"question": question}, return_only_outputs=True)
            return results

        return [
            Tool(
                args_schema=DocumentInput,
                name=document_name,
                description=f"useful when you want to answer questions about {document_name}",
                func=run_chain,
                return_direct=False,
            )
            for document_name in document_names
        ]

    def get_citations_from_source_documents(self, prompt_result: dict[str, Any]) -> list[BaseCitation]:
        source_documents = [
            source_doc
            for intermediate_step in prompt_result["intermediate_steps"]
            for source_doc in intermediate_step[1]["source_documents"]
        ]
        citations: Citations = Citations(set(), False)
        citations.get_unique_citations(source_documents)
        citations_list = list(citations.citations)
        return citations_list

    def add_to_chat_history(self, chat_history: list[str], prompt: str, answer: str, citations: list[Citation]) -> None:
        """
        Method to add chat history
        """
        self.chat_history_internal.append(chat_history)
        self.chat_history_export.append((prompt, answer, citations))

    def send_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Method to send a prompt to the chatbot
        """
        result = self.agent(inputs={"input": prompt, "agent_history": self.chat_history_internal})
        citations: list[Citation] = self.get_citations_from_source_documents(result)
        self.add_to_chat_history(result["agent_history"], prompt, result["output"], citations)
        response = {
            "answer": result["output"],
            "citations": citations,
            "chat_history": self.chat_history_export,
        }
        return response
