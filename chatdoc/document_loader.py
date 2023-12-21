"""
Module to load documents in several formats
"""
import itertools
from typing import Iterator, Any, cast
from pathlib import Path


from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.schema import Document

FilePath = str | Path | None


class DocumentLoader:
    """
    Generic base class to load documents
    """

    @staticmethod
    def __initialize_document_loader(abs_file_path: str, file_extension: str) -> BaseLoader:
        match file_extension:
            case ".pdf":
                return PyPDFLoader(abs_file_path)
            case ".docx":
                return Docx2txtLoader(abs_file_path)
            case _:
                raise NotImplementedError(f"A file loader for {file_extension} has not been implemented yet")

    def __init__(
        self,
        model_name: str,
        file_paths: list[Path],
        chunk_size: int = 1000,
        **kwargs: dict[str, Any],
    ):
        self.file_paths: list[Path] = file_paths
        loaders: list[BaseLoader] = [self.__initialize_document_loader(
            abs_file_path=str(file_path_obj.absolute()),
            file_extension=file_path_obj.suffix,
        ) for file_path_obj in self.file_paths]
        self._document_iterator: Iterator[Document] = itertools.chain(*[loader.lazy_load() for loader in loaders])
        self._text_splitter: TextSplitter = cast(
            TextSplitter,
            kwargs.get(
                "text_splitter",
                TokenTextSplitter(model_name=model_name, chunk_size=cast(int, kwargs.get("chunk_size", chunk_size)), chunk_overlap=0),
            ),
        )

    @property
    def split_data(self) -> list[Document]:
        """
        Obtain the split data from the document loader
        """
        return self._text_splitter.split_documents(self._document_iterator)

    def get_file_names(self, with_extension: bool = True) -> list[str]:
        """
        Obtain the file name of the document with or without the file extension
        """
        if with_extension:
            return [file_path_obj.name for file_path_obj in self.file_paths]
        return [file_path_obj.stem for file_path_obj in self.file_paths]

    def get_file_extensions(self) -> list[str]:
        """
        Obtain the file extension
        """
        return [file_path_obj.suffix for file_path_obj in self.file_paths]
