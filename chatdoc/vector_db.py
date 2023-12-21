"""
Module definine the VectorDatabase class
"""
from pathlib import Path
import uuid

from langchain.embeddings.base import Embeddings

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma as ChromaDB

from .document_loader import DocumentLoader


class VectorDatabase:
    """
    The VectorDatabase class that creates a ChromaDB store locally
    """

    def __init__(self, document_loader: DocumentLoader, embeddings: Embeddings, persist: bool = False) -> None:
        self.collection_name: str = str(uuid.uuid4())
        self.local_directory_name: str = self.collection_name + "_embedding"
        self.persist_directory: Path | None = None
        self.embeddings_instance: Embeddings = embeddings
        self.vector_store: VectorStore = ChromaDB.from_documents(
            documents=document_loader.split_data,
            embedding=self.embeddings_instance,
            collection_name=self.collection_name,
        )
        if persist:
            self.persist_directory = Path.cwd() / "embeddings" / self.local_directory_name
            self.vector_store.__setattr__("_persist_directory", self.persist_directory.absolute())
            self.vector_store.persist()
