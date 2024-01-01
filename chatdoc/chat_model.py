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
                if self.api_key is None:
                    self.api_key = Utils.get_env_variable("OPENAI_API_KEY")
                return ChatOpenAI(api_key=self.api_key, model=self.chat_model_name)
            case "huggingface":
                if self.api_key is None:
                    self.api_key = Utils.get_env_variable("HUGGINGFACE_API_KEY")
                raise NotImplementedError("HuggingFace chat model not implemented")
            case _:
                raise ValueError("Invalid vendor name")

    def __init__(
        self,
        chat_model_vendor_name: str | None = None,
        chat_model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initializes the ChatModel object by setting
        the chat model.
        """
        self.vendor_name: str = (
            chat_model_vendor_name
            if chat_model_vendor_name is not None
            else Utils.get_env_variable("CHAT_MODEL_VENDOR_NAME")
        )
        self.chat_model_name = (
            chat_model_name if chat_model_name is not None else Utils.get_env_variable("CHAT_MODEL_NAME")
        )
        self.api_key = api_key
        self.chat_model: BaseChatModel = self._load_chat_model()
