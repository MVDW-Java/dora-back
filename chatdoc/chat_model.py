from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from .utils import Utils

class ChatModel:
    """
    A class representing a chat model.

    Attributes:
        chat_model (object): The chat model.

    Methods:
        __init__(): Initializes the ChatModel object by setting
          the chat model.
    """

    def _set_vendor_name(self):
        """
        Sets the vendor name based on the environment variable.
        """
        self.vendor_name = Utils.get_env_variable("CHAT_MODEL_VENDOR_NAME")

    def _set_chat_model_name(self):
        """
        Sets the embedding model name based on the environment variable.
        """
        self.chat_model_name = Utils.get_env_variable("CHAT_MODEL_NAME")

    def _load_chat_model(self) -> BaseChatModel:
        """
        Loads the embedding function based on the vendor name.

        Returns:
            object: The embedding function.

        Raises:
            ValueError: If the vendor name is invalid.
        """
        match self.vendor_name:
            case "openai":
                return ChatOpenAI(
                   api_key=Utils.get_env_variable("OPENAI_API_KEY"), model=self.chat_model_name)
            case "huggingface":
                raise NotImplementedError("HuggingFace chat model not implemented")
            case _:
                raise ValueError("Invalid vendor name")

    def __init__(self):
        """
        Initializes the ChatModel object by setting
        the chat model.
        """
        self._set_vendor_name()
        self._set_chat_model_name()
        self.chat_model: BaseChatModel = self._load_chat_model()
