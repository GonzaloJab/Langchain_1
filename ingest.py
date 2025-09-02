from dotenv import load_dotenv
import os
load_dotenv()

# from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
#Import ollama embeddings
from langchain_ollama import OllamaEmbeddings
if __name__ == "__main__":
    

    # embeddings = OllamaEmbeddings(model="phi3.5:3.8b")

    # 1. Load the documents
    print("Preparing data...")
    loader = TextLoader(r"E:\langchain-course\medium-blog1.txt", encoding="utf-8")
    documents = loader.load()
    # print(documents)

    # 2. Split the documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    # print(chunks)

    # 3. Embed the chunks
    print("Embedding chunks...")
    embeddings = OllamaEmbeddings(model="phi3.5:3.8b")
    

    # 4. Add the chunks to the vector store
    print("Adding chunks to the vector store...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))