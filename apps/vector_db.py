import numpy as np
from faiss import IndexFlatL2
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, documents):
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = encoder.encode([doc.page_content for doc in documents])
        embeddings = np.array(embeddings, dtype=np.float32) #Converting into FAISS valid format
        self.vector_index = IndexFlatL2(embeddings.shape[1])
        self.vector_index.add(embeddings)
        self.index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        self.docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
    
    def create_vector_db(self):
        vector_store = FAISS(index=self.vector_index, 
                             embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                             docstore=self.docstore,
                             index_to_docstore_id=self.index_to_docstore_id)
        
        return vector_store

if __name__ == "__main__":
    pass