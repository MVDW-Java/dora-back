from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from .utils import Utils

class Embedding:
    """
    A class representing an embedding model.

    Attributes:
        vendor_name (str): The name of the embedding vendor.
        embedding_model_name (str): The name of the embedding model.
        embedding_function (object): The embedding function.

    Methods:
        _get_env_variable(variable_name: str) -> str: Retrieves the value of an environment variable.
        _set_vendor_name(): Sets the vendor name based on the environment variable.
        _set_embedding_model_name(): Sets the embedding model name based on the environment variable.
        _load_embedding_function() -> object: Loads the embedding function based on the vendor name.
        __init__(): Initializes the Embedding object by setting
          the vendor name, embedding model name, and loading the embedding function.
    """

    def _set_vendor_name(self):
        """
        Sets the vendor name based on the environment variable.
        """
        self.vendor_name = Utils.get_env_variable("EMBEDDING_VENDOR_NAME")

    def _set_embedding_model_name(self):
        """
        Sets the embedding model name based on the environment variable.
        """
        self.embedding_model_name = Utils.get_env_variable("EMBEDDING_MODEL_NAME")

    def _load_embedding_function(self) -> Embeddings:
        """
        Loads the embedding function based on the vendor name.

        Returns:
            object: The embedding function.

        Raises:
            ValueError: If the vendor name is invalid.
        """
        match self.vendor_name:
            case "openai":
                return OpenAIEmbeddings(
                   api_key=Utils.get_env_variable("OPENAI_API_KEY"), model=self.embedding_model_name)
            case "huggingface":
                return HuggingFaceEmbeddings(
                    api_key=Utils.get_env_variable("HUGGINGFACE_API_KEY"), model=self.embedding_model_name)
            case _:
                raise ValueError("Invalid vendor name")

    def __init__(self):
        """
        Initializes the Embedding object by setting
          the vendor name, embedding model name, and loading the embedding function.
        """
        self._set_vendor_name()
        self._set_embedding_model_name()
        self.embedding_function = self._load_embedding_function()
        