import pytest
from langchain.chat_models.openai import ChatOpenAI
from chatdoc.chat_model import ChatModel


@pytest.fixture()
def openai_chat_model(monkeypatch):
    """
    Function to test the chat model with OpenAI vendor.

    Args:
        monkeypatch: The monkeypatch fixture.

    Returns:
        ChatModel: An instance of the ChatModel class.

    """
    monkeypatch.setenv('CHAT_MODEL_VENDOR_NAME', 'openai')
    monkeypatch.setenv('CHAT_MODEL_NAME', 'gpt-3')
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    yield ChatModel

@pytest.fixture()
def huggingface_chat_model(monkeypatch):
    """
    A test function that sets the environment variable 'CHAT_MODEL_VENDOR_NAME' to 'huggingface'
    and yields the ChatModel object.

    Args:
        monkeypatch: A pytest fixture that allows modifying environment variables during testing.

    Yields:
        ChatModel: An instance of the ChatModel class.

    """
    monkeypatch.setenv('CHAT_MODEL_VENDOR_NAME', 'huggingface')
    monkeypatch.setenv('CHAT_MODEL_NAME', 'BloombergGPT')
    yield ChatModel

@pytest.fixture()
def invalid_vendor_chat_model(monkeypatch):
    """
    Test case for ChatModel with an invalid vendor name.
    
    Args:
        monkeypatch: A pytest fixture used to modify environment variables.
    
    Yields:
        ChatModel: An instance of the ChatModel class.
    """
    monkeypatch.setenv('CHAT_MODEL_VENDOR_NAME', 'mistral')
    monkeypatch.setenv('CHAT_MODEL_NAME', 'mistral-7B')
    yield ChatModel


def test_set_vendor_name(openai_chat_model): # pylint: disable=W0621
    """
    Test case to verify the `vendor_name` property of the `chat_model_openai` object.

    Args:
        monkeypatch: Monkeypatch fixture for patching objects during testing.
        chat_model_openai: Instance of the `ChatModel` class with the 'openai' vendor name.

    Returns:
        None
    """
    assert openai_chat_model().vendor_name == 'openai'

def test_set_chat_model_name(openai_chat_model): # pylint: disable=W0621
    """
    Test case to verify if the chat model name is set correctly.
    
    Args:
        chat_model_openai: An instance of the ChatModelOpenAI class.
    
    Returns:
        None
    """
    assert openai_chat_model().chat_model_name == 'gpt-3'

def test_load_chat_model_openai(openai_chat_model): # pylint: disable=W0621
    """
    Test case for loading the chat model from OpenAI.

    Args:
        chat_model_openai (MagicMock): The chat model object.

    Returns:
        None
    """
    assert isinstance(openai_chat_model().chat_model, ChatOpenAI)

def test_load_chat_model_huggingface(huggingface_chat_model): # pylint: disable=W0621
    """
    Test case for the _load_chat_model method of the chat_model_huggingface object.
    It checks if a NotImplementedError is raised when calling the _load_chat_model method.
    """
    with pytest.raises(NotImplementedError):
        huggingface_chat_model()
        

def test_load_chat_model_invalid_vendor(invalid_vendor_chat_model): # pylint: disable=W0621
    """
    Test case to ensure that loading a chat model with an invalid vendor raises a ValueError.
    """
    with pytest.raises(ValueError):
        invalid_vendor_chat_model()