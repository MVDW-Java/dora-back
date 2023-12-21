class PaperQABot:
    """
    The chatbot that uses the paperQA library
    """
    def __init__(self, file_paths: list[str | Path] | None = None,
        chat_model_name: str = "gpt-3.5-turbo",
        embedding_model_name: str = "text-embedding-ada-002",
        **kwargs: dict[str, Any]) -> None:
        if file_paths is None:
            self.file_paths: list[Path] = [Path.cwd() / Path(os.environ["FILE_PATH"])]
        else:
            self.file_paths: list[Path] = [Path(file_path) for file_path in file_paths]
        if "embeddings" in kwargs:
            self.embeddings = kwargs["embeddings"]
        else:
            self.embeddings = OpenAIEmbeddings(api_key=Utils.read_api_key(kwargs), model=embedding_model_name)
        self.docs = Docs(llm=chat_model_name, embeddings=self.embeddings)
        for file_path in self.file_paths:
            self.docs.add(file_path)    

    def run(self, text_width: int = 20):
        """
        Method to run the chatboat using input and printing
        """
        qry = ""
        while qry != "done":
            qry = input("Question: ")
            start = time.time()
            print(f"Initial question:\n {qry:<{text_width}}", flush=True)
            if qry != "done":
                response = self.docs.query(qry)
                # print(f"Rephrased question:\n {response.question:<{text_width}}", flush=True)
                print(f"Answer:\n {response.formatted_answer:<{text_width}}", flush=True)
                print("SOURCES: ", flush=True)
                print(response.references, flush=True)
            end = time.time()
            print(f"Time: {end - start:.2f}s", flush=True)