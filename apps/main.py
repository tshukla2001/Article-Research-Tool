import streamlit as st
import time
import os
import pickle
from model import Model
from url_retriever import URLRetriever
from vector_db import VectorDB
from chain import Chain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from faiss import IndexFlatL2
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain


model = Model("llama-3.3-70b-versatile")
model = model.call()

st.title("News Research Tool 📰🗞️")
st.write("This tool helps you to search for specific topics from the given links.")

st.sidebar.title("News Article URLs")
st.sidebar.write("Add the links of the articles you want to search in.")

a1 = st.sidebar.text_input("Article 1", key="article1")
a2 = st.sidebar.text_input("Article 2", key="article2")
a3 = st.sidebar.text_input("Article 3", key="article3")
process_articles = st.sidebar.button("Process Articles")

main_placeholder = st.empty()

if process_articles:
    loader = UnstructuredURLLoader(
            urls=[a1, a2, a3]
        )
    
    main_placeholder.write("Loading articles...✅✅✅")

    loaded_articles = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300
        )
    
    main_placeholder.write("Creating Chunks...✅✅✅")

    docs = text_splitter.split_documents(loaded_articles)

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode([doc.page_content for doc in docs])
    embeddings = np.array(embeddings, dtype=np.float32) #Converting into FAISS valid format
    vector_index = IndexFlatL2(embeddings.shape[1])
    vector_index.add(embeddings)
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

    vector_store = FAISS(index=vector_index, 
                             embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                             docstore=docstore,
                             index_to_docstore_id=index_to_docstore_id)

    main_placeholder.write("Started Building Vector Database...✅✅✅")
    time.sleep(2)

    with open("vector_db.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    
    main_placeholder.write("Vector database saved...✅✅✅")

query = main_placeholder.text_input("Enter your query here:")

if query:
    if os.path.exists("vector_db.pkl"):
        with open("vector_db.pkl", "rb") as f:
            vector_store = pickle.load(f)
        
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm = model,
            retriever = vector_store.as_retriever()
        )

        main_placeholder.write("Running the chain...✅✅✅")
        time.sleep(2)

        result = chain({'question': query}, return_only_outputs=True)

        main_placeholder.empty()

        st.header("Results:")
        st.write(result['answer'])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            unique_sources = set(sources_list)
            for source in unique_sources:
                st.write(source)
        
        main_placeholder.text_input("Enter your query here:", key="new_query")

