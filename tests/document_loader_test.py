import itertools
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chatdoc.doc_loader.document_loader import DocumentLoader, SupportedLoader
from chatdoc.doc_loader.loader_factory import LoaderFactory


def test_document_loader():
    # Mock document_dict
    document_dict = {
        "doc1": Path("/path/to/doc1.txt"),
        "doc2": Path("/path/to/doc2.txt"),
    }

    # Mock LoaderFactory
    mock_loader_factory = MagicMock(spec=LoaderFactory)
    mock_loader_factory.create.return_value = SupportedLoader

    # Initialize DocumentLoader
    document_loader = DocumentLoader(document_dict, mock_loader_factory)

    # Test initialize_loaders
    loaders = document_loader.initialize_loaders(document_dict)
    for loader in loaders:
        assert isinstance(loader, SupportedLoader), "All loaders should be instances of SupportedLoader"

    # Test chain_document_iterators
    chained_iterators = document_loader.chain_document_iterators()
    assert isinstance(chained_iterators, itertools.chain), "Should return an instance of itertools.chain"
