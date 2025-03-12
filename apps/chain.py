from langchain.chains import RetrievalQAWithSourcesChain

class Chain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def create_chain(self):
        self.chain = RetrievalQAWithSourcesChain.from_llm(
            llm = self.llm,
            retriever = self.retriever.as_retriever()
        )

        return self.chain
    
    def run_chain(self, query):
        return self.chain.invoke({'question': query})

if __name__ == "__main__":
    pass