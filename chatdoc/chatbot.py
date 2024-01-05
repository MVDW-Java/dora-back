from typing import Any
from logging import Logger
import time
from pathlib import Path

from langchain.tools.vectorstore.tool import VectorStoreQAWithSourcesTool
from langchain.tools import Tool
from langchain.agents import Agent, initialize_agent, AgentType, OpenAIFunctionsAgent, ConversationalChatAgent
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, RetrievalQA
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field

from chatdoc.vector_db import VectorDatabase
from chatdoc.citation import Citations, BaseCitation
from chatdoc.embed.embedding_factory import EmbeddingFactory
from chatdoc.chat_model import ChatModel


class DocumentInput(BaseModel):
    question: str = Field()


# class DocumentInput(BaseModel):
#     question: str = Field()
#     chat_history: list = Field()


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
        # self._agent = OpenAIFunctionsAgent.from_llm_and_tools(
        #     llm=self._chat_model,
        #     tools=self.tools,
        #     memory=self._memory,
        #     verbose=True,
        #     output
        # )
        self.agent = initialize_agent(
            agent=AgentType.OPENAI_FUNCTIONS,
            tools=self.tools,
            llm=self._chat_model,
            memory=self.agent_memory,
            verbose=True,
            return_intermediate_steps=True,
            return_source_documents=True,
            # handle_passing_errors=True,
        )
        self.logger = logger if logger else Logger("chatbot")
        self.chat_history_internal: list = []
        self.chat_history_export: list[tuple[str, str, list[BaseCitation]]] = []

    def create_tools(self, document_dict: dict[str, str]) -> list[Tool]:
        document_names = [str(Path(document_name).stem) for document_name in document_dict.keys()]
        # retrieval_chain = ConversationalRetrievalChain.from_llm(
        #     llm=self._chat_model,
        #     retriever=self._vector_database.retriever,
        #     memory=self._memory,
        #     return_source_documents=True,
        #     verbose=True,
        # )
        retrieval_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=self._chat_model,
            retriever=self._vector_database.retriever,
            memory=self.chain_memory,
            return_source_documents=True,
            verbose=True,
        )

        def run_chain(question: str):
            results = retrieval_chain({"question": question}, return_only_outputs=True)
            # return str(results)
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

    def send_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Method to send a prompt to the chatbot
        """
        result = self.agent(inputs={"input": prompt, "agent_history": self.chat_history_internal})
        print(result)
        # citations: Citations = Citations(set(), False)
        # citations.get_unique_citations(result["source_documents"])
        # citations_list = list(citations.citations)
        # self.chat_history_internal.append(result["chat_history"])
        # self.chat_history_export.append((prompt, result["answer"], citations_list))
        response = {
            # "answer": result["answer"],
            # "citations": citations_list,
            # "chat_history": self.chat_history_export,
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
                # response = self._retrieval_chain({"question": qry, "chat_history": chat_history})
                response = self.agent(inputs={"input": qry})
                print(f"Answer:\n {response['answer']:<{text_width}}", flush=True)
                print("SOURCES: ", flush=True)
                citations: Citations = Citations(set(), with_proof)
                citations.get_unique_citations(response["source_documents"])
                citations.print_citations()
            end = time.time()
            print(f"Time: {end - start:.2f}s", flush=True)
