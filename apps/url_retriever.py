from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class URLRetriever:
    def __init__(self, url1, url2, url3):
        self.loader = UnstructuredURLLoader(
            urls=[url1, url2, url3]
        )
    
    def load(self):
        return self.loader.load()
    
    def doc_split_into_chunks(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300
        )

        docs = text_splitter.split_documents(docs)
        return docs

if __name__ == "__main__":
    retriever = URLRetriever(
        "https://www.bbc.com/news/articles/cz7vlezv05no",
        "https://timesofindia.indiatimes.com/business/india-business/stock-market-today-bse-sensex-nifty50-march-06-2025-dalal-street-indian-equities-global-markets-trump-tariff/articleshow/118748289.cms",
        "https://www.thehindu.com/business/markets/rupee-falls-6-paise-to-settle-at-8712-against-us-dollar/article69298165.ece"
    )
    
    article_result = retriever.load()
    # print(article_result[0].page_content)

    docs = retriever.doc_split_into_chunks(article_result)
    print(docs[5].page_content)