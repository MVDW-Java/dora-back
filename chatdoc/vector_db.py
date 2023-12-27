"""
Module definine the VectorDatabase class
"""
import os
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from chromadb import PersistentClient
from chromadb.api import ClientAPI


class VectorDatabase:
    """
    The VectorDatabase class that creates a ChromaDB store locally
    """
    _instance = None

    _chroma_db_client = None

    @property
    def chroma_client(self) -> ClientAPI:
        """
        ChromaDB client
        """
        if self._chroma_db_client is not None:
            return self._chroma_db_client
        dora_env = os.environ.get('DORA_ENV')
        match dora_env:
            case 'DEV', 'TST':
                self._chroma_db_client = PersistentClient()
            case 'PROD':
                # Connect to ChromaDB in the cloud
                # Add code here to connect to ChromaDB in the cloud
                raise NotImplementedError("ChromaDB in the cloud not implemented yet")
            case _:
                raise ValueError("Invalid environment")
        return self._chroma_db_client


    def __init__(self, collection_name: str, embedding_fn: Embeddings) -> None:
        self.collection_name = collection_name
        self.chroma_instance = Chroma(
            collection_name=collection_name,
            client=self.chroma_client,
            embedding_function=embedding_fn,
        )
        self.retriever = self.chroma_instance.as_retriever()

    async def add_documents(self, documents: list[Document]) -> None:
        """
        Add multiple documents to the vector database.

        Args:
            documents (list[Document]): 
                A list of Document objects to be added when this endpoint is called from `server.py`.

        Returns:
            None
        """
        await self.chroma_instance.aadd_documents(documents)
