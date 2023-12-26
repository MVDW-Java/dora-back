"""
Module to load documents in several formats
"""
import itertools
from typing import Iterator, Any
from pathlib import Path
import os


from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.schema import Document

FilePath = str | Path | None


class DocumentLoader:
    """
    Generic base class to load documents.

    This class provides functionality to load documents of different file types
    and split them into smaller chunks for further processing.

    Attributes:
        document_iterator (Iterator[Document]): An iterator that yields Document objects.
        text_splitter (TextSplitter): An instance of TextSplitter used to split the documents into smaller chunks.
    """

    @staticmethod
    def __initialize_document_loader(abs_file_path: str, file_extension: str) -> BaseLoader:
        """
        Initialize and return the appropriate document loader based on the file extension.

        Args:
            abs_file_path (str): The absolute file path of the document.
            file_extension (str): The file extension of the document.

        Returns:
            BaseLoader: An instance of the appropriate document loader.

        Raises:
            NotImplementedError: If a file loader for the given file extension has not been implemented yet.
        """
        match file_extension:
            case ".pdf":
                return PyPDFLoader(abs_file_path)
            case ".docx":
                return Docx2txtLoader(abs_file_path)
            case _:
                raise NotImplementedError(f"A file loader for {file_extension} has not been implemented yet")

    def __load_token_text_splitter(self, **kwargs) -> TextSplitter:
        """
        Load and return the TextSplitter instance.

        If the chunk size is specified in the environment variable "CHUNK_SIZE",
        it will be used. Otherwise, if the chunk size is specified in the kwargs,
        it will be used. Otherwise, a default chunk size of 1000 will be used.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            TextSplitter: An instance of TextSplitter.

        """
        if "CHUNK_SIZE" in os.environ:
            chunk_size = int(os.environ["CHUNK_SIZE"])
        elif "chunk_size" in kwargs:
            chunk_size = int(kwargs["chunk_size"])
        else:
            print("No chunk size specified, defaulting to 1000") # TODO: replace with logging
            chunk_size = 1000
        return kwargs.get(
                "text_splitter",
                TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0),
            )

    def __init__(
        self,
        document_dict: dict[str, Path],
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the DocumentLoader instance.

        Args:
            document_dict (dict[str, Path]): A dictionary mapping document names to their corresponding file paths.
            **kwargs: Additional keyword arguments.
        """
        loaders: list[BaseLoader] = [self.__initialize_document_loader(
            abs_file_path=str(file_path_obj.absolute()),
            file_extension=file_path_obj.suffix,
        ) for file_path_obj in document_dict.values()]
        self.document_iterator: Iterator[Document] = itertools.chain(*[loader.lazy_load() for loader in loaders])
        self.text_splitter: TextSplitter = self.__load_token_text_splitter(**kwargs)

    # @property
    # def split_data(self) -> list[Document]:
    #     """
    #     Obtain the split data from the document loader.

    #     Returns:
    #         list[Document]: A list of Document objects representing the split data.
    #     """
    #     return self._text_splitter.split_documents(self._document_iterator)
