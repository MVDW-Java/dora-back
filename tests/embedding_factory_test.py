import pytest

from chatdoc.embed.embedding_factory import EmbeddingFactory


def test_env_fn_called_missing_vendor_name():
    """
    Test case to ensure that the env_fn is called when the vendor name is not set.
    """
    with pytest.raises(ValueError, match="not set in environment"):
        EmbeddingFactory(embedding_model_name="gpt-3")


def test_env_fn_called_missing_model_name():
    """
    Test case to ensure that the env_fn is called when the model name is not set.
    """
    with pytest.raises(ValueError, match="not set in environment"):
        EmbeddingFactory(vendor_name="openai")


def test_api_key_missing():
    """
    Test case to ensure that an error is raised when the API key is missing.
    """
    with pytest.raises(ValueError, match="not set in environment"):
        EmbeddingFactory(vendor_name="openai", embedding_model_name="gpt-3").create()


def test_unknown_vendor_name():
    """
    Test case to ensure that an error is raised when the vendor name is unknown.
    """
    with pytest.raises(ValueError, match="No embedding available for vendor name"):
        EmbeddingFactory(vendor_name="unknown", embedding_model_name="gpt-3").create()
