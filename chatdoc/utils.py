import os

class Utils:
    @classmethod
    def get_env_variable(cls, variable_name: str) -> str:
        """
        Retrieves the value of an environment variable.

        Args:
            variable_name (str): The name of the environment variable.

        Returns:
            str: The value of the environment variable.

        Raises:
            ValueError: If the environment variable is not set.
        """
        if variable_name not in os.environ:
            raise ValueError(f"{variable_name} not set in environment")
        else:
            return os.environ[variable_name]