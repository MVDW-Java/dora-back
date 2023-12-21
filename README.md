# rag-test
This is a repository that implements RAG using GPT-3.5

## How to run the code
Either clone this project in VSCode or open a new codespace (if you have not been invited into another one).

The `devcontainer.json` should contain all the plugins needed to get going including the installation of Poetry (which may need to be done manually).

Additionally, set the `FILE_PATH` environment variable to where you store the PDF file and include your OpenAI API key in the `OPENAI_API_KEY` environment variable.

Subsequently, run `poetry update` in the terminal to install all the dependencies and create the environment. 

Moreover, you need to run `poetry run ipython kernel install --user --name=<NAME>`; for `NAME`, you can pick any name for the Jupyter Notebook kernel.

Lastly, you can run the Jupyter notebook inside the Poetry environment and do not forget to select the created kernel.

