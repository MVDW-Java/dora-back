from unittest.mock import MagicMock
import pytest
from chatdoc.chat_model import ChatModel

def test_set_vendor_name(monkeypatch):
    monkeypatch.setattr('chatdoc.chat_model.Utils.get_env_variable', lambda _: 'openai')
    chat_model = ChatModel()
    assert chat_model.vendor_name == 'openai'

def test_set_chat_model_name(monkeypatch):
    monkeypatch.setattr('chatdoc.chat_model.Utils.get_env_variable', lambda _: 'gpt-3')
    chat_model = ChatModel()
    assert chat_model.chat_model_name == 'gpt-3'

def test_load_chat_model_openai(monkeypatch):
    monkeypatch.setattr('chatdoc.chat_model.Utils.get_env_variable', lambda _: 'openai')
    monkeypatch.setattr('chatdoc.chat_model.ChatOpenAI', MagicMock())
    chat_model = ChatModel()
    assert isinstance(chat_model.chat_model, MagicMock)

def test_load_chat_model_huggingface(monkeypatch):
    monkeypatch.setattr('chatdoc.chat_model.Utils.get_env_variable', lambda _: 'huggingface')
    with pytest.raises(NotImplementedError):
        ChatModel()

def test_load_chat_model_invalid_vendor(monkeypatch):
    monkeypatch.setattr('chatdoc.chat_model.Utils.get_env_variable', lambda _: 'invalid_vendor')
    with pytest.raises(ValueError):
        ChatModel()