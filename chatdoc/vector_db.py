"""
Module definine the VectorDatabase class
"""
from pathlib import Path
import uuid
import tempfile
import chromadb
from typing import Any
from langchain.embeddings.base import Embeddings

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma as ChromaDB

from .document_loader import DocumentLoader




class VectorDatabase:
    """
    The VectorDatabase class that creates a ChromaDB store locally
    """
    _chroma_client = chromadb.PersistentClient()

    @property
    def chroma_client(self) -> chromadb.PersistentClient:
        """
        ChromaDB client
        """
        return self._chroma_client
    

    def __init__(self, collection_name: str, document_dict: dict[str, Any], document_loader: DocumentLoader) -> None:
        current_collection = self.chroma_client.get_or_create_collection(collection_name)
        
        
