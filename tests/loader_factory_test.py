from unittest.mock import patch, create_autospec
from collections.abc import Iterator
import pytest
from chatdoc.doc_loader.loader_factory import LoaderFactory, PyPDFLoader, Docx2txtLoader


@pytest.fixture(name="loader_factory")
def loader_factory_fixture() -> type[LoaderFactory]:
    """
    Fixture function that returns the LoaderFactory class.

    Returns:
        type[LoaderFactory]: The LoaderFactory class.
    """
    return LoaderFactory


@pytest.fixture(name="loader_factory_pdf")
def loader_factory_pdf_fixture(loader_factory) -> Iterator[LoaderFactory]:
    """
    Fixture for testing the loader_factory function with PyPDFLoader.

    Args:
        loader_factory: The loader_factory function to be tested.

    Yields:
        LoaderFactory: The result of calling the loader_factory function.
    """
    mock_loader = create_autospec(PyPDFLoader, instance=True)
    with patch("chatdoc.doc_loader.loader_factory.PyPDFLoader", return_value=mock_loader):
        yield loader_factory()


@pytest.fixture(name="loader_factory_docx")
def loader_factory_docx_fixture(loader_factory) -> Iterator[LoaderFactory]:
    """
    Fixture for testing the loader_factory function with Docx2txtLoader.

    Args:
        loader_factory: The loader_factory function to be tested.

    Yields:
        LoaderFactory: The loader_factory function called with the mocked Docx2txtLoader.
    """
    mock_loader = create_autospec(Docx2txtLoader, instance=True)
    with patch("chatdoc.doc_loader.loader_factory.Docx2txtLoader", return_value=mock_loader):
        yield loader_factory()


def test_create_pdf_loader(loader_factory_pdf):
    """
    Test case for creating a PDF loader.
    """
    dummy_file_path = "/dummy/path/file.pdf"
    loader = loader_factory_pdf.create(dummy_file_path, ".pdf")
    assert isinstance(loader, PyPDFLoader)


def test_create_word_loader(loader_factory_docx):
    """
    Test case for creating a PDF loader.
    """
    dummy_file_path = "/dummy/path/file.docx"
    loader = loader_factory_docx.create(dummy_file_path, ".docx")
    assert isinstance(loader, Docx2txtLoader)
