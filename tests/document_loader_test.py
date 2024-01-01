from itertools import chain
from pathlib import Path
from unittest.mock import MagicMock
from langchain.schema.document import Document

import pytest

from chatdoc.doc_loader.document_loader import DocumentLoader
from chatdoc.doc_loader.document_loader_factory import DocumentLoaderFactory, BaseLoader


@pytest.fixture(name="mock_loader_factory")
def fixture_mock_loader_factory():
    """
    Returns a mock loader factory object that creates a mock document loader.

    Returns:
        MagicMock: A mock loader factory object.
    """
    mock_loader_factory = MagicMock(spec=DocumentLoaderFactory)
    mock_doc_loader = MagicMock(spec=BaseLoader)
    mock_loader_factory.create.return_value = mock_doc_loader
    return mock_loader_factory


@pytest.fixture(name="mock_document")
def fixture_mock_document():
    """
    Creates and returns a mock document object.

    Returns:
        MagicMock: A mock object with the Document specification.
    """
    mock_document = MagicMock(spec=Document)
    return mock_document


@pytest.fixture(name="document_loader")
def fixture_document_loader(mock_loader_factory):
    """
    Fixture function that creates a DocumentLoader instance with a dictionary of document paths.

    Args:
        mock_loader_factory: A mock loader factory object.

    Returns:
        A DocumentLoader instance.

    """
    document_dict = {
        "doc1": Path("/path/to/doc1.docx"),
        "doc2": Path("/path/to/doc2.pdf"),
    }
    document_loader = DocumentLoader(document_dict, mock_loader_factory)
    return document_loader


@pytest.fixture(name="mock_document_loader")
def fixture_mock_document_loader(mock_document):
    """
    Creates a mock DocumentLoader object with a chain of document iterators
    that returns the provided mock_document.

    Args:
        mock_document: The mock document to be returned by the document iterators.

    Returns:
        A MagicMock object representing the mock DocumentLoader.
    """
    mock_document_loader = MagicMock(spec=DocumentLoader)
    mock_document_loader.chain_document_iterators.return_value = iter([mock_document])
    return mock_document_loader


def test_document_loader_iterator(document_loader):
    """
    Test case to verify the document_loader's document_iterator property.

    Args:
        document_loader: An instance of the document_loader class.

    Raises:
        AssertionError: If the document_loader's document_iterator is not an instance of itertools.chain.
    """
    assert isinstance(document_loader.document_iterator, chain), "Should return an instance of itertools.chain"


def test_document_iterator_document(mock_document_loader):
    """
    Test that the document iterator returns an instance of Document.
    """
    document_iterator = mock_document_loader.chain_document_iterators()
    assert isinstance(next(document_iterator), Document), "Should return an instance of Document"
