import os
import nltk
from dotenv import load_dotenv
from langchain_groq import ChatGroq

nltk.download('averaged_perceptron_tagger')

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = API_KEY
    
    def call(self):
        return ChatGroq(
            model=self.model_name,
            temperature=0.3,
            api_key=self.api_key
        )


if __name__ == "__main__":
    model = Model(
        model_name = "llama-3.3-70b-versatile"
    )
    model = model.call()
    
    sample_result = model.invoke("What is the capital of France?")
    print(sample_result.content)
