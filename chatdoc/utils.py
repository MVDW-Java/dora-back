from os import environ as env
from typing import Any


class Utils:
    """
    A class with utility methods
    """

    @classmethod
    def get_openai_api_key(cls, kwargs: dict[str, Any] | None = None) -> str:
        """
        Read the API key from the environment or from the kwargs
        """
        openai_api_key: str = kwargs.get("openai_api_key", env["OPENAI_API_KEY"]) if kwargs is not None else env["OPENAI_API_KEY"]
        return openai_api_key
