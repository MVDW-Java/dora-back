from langchain.schema.embeddings import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ..utils import Utils


class EmbeddingFactory:
    """
    Factory class for creating different types of embeddings.

    Attributes:
        embedding_map (dict[str, type[Embeddings]]): A mapping of vendor names to embedding classes.
        api_key_map (dict[str, str]): A mapping of vendor names to API key environment variable names.

    Methods:
        create(vendor_name: str, embedding_model_name: str, api_key: str | None = None) -> Embeddings:
            Creates an instance of the specified embedding class.

    """

    def __init__(self, vendor_name: str | None = None, embedding_model_name: str | None = None) -> None:
        """
        Initializes an instance of the EmbeddingFactory class.

        Args:
            vendor_name (str | None, optional): The name of the vendor. Defaults to None.
            embedding_model_name (str | None, optional): The name of the embedding model. Defaults to None.

        """
        self.embedding_map: dict[str, type[Embeddings]] = {
            "openai": OpenAIEmbeddings,
            "huggingface": HuggingFaceEmbeddings,
        }
        self.api_key_map: dict[str, str] = {
            "openai": "OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }
        self.vendor_name = vendor_name if vendor_name is not None else Utils.get_env_variable("EMBEDDING_VENDOR_NAME")
        self.embedding_model_name = (
            embedding_model_name if embedding_model_name is not None else Utils.get_env_variable("EMBEDDING_MODEL_NAME")
        )

    def create(self, api_key: str | None = None) -> Embeddings:
        """
        Creates an instance of the specified embedding class.

        Args:
            api_key (str | None, optional): The API key to be used. Defaults to None.

        Returns:
            Embeddings: An instance of the specified embedding class.

        Raises:
            ValueError: If no embedding is available for the specified vendor name.

        """
        embedding_class = self.embedding_map.get(self.vendor_name)
        if embedding_class is None:
            raise ValueError(f"No embedding available for vendor name {self.vendor_name}")
        if api_key is None:
            api_key_var = self.api_key_map.get(self.vendor_name)
            if api_key_var is None:
                raise ValueError(f"No API key environment variable available for vendor name {self.vendor_name}")
            api_key = Utils.get_env_variable(api_key_var)
        return embedding_class(api_key=api_key, model=self.embedding_model_name, disallowed_special=())
