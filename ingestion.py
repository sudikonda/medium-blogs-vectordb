import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingestion started")

    loader = TextLoader(
        os.getenv("MEDIUM_BLOGS_FILE_PATH"),
        encoding="utf-8"
    )

    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("Splitting text into chunks")

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("Creating Pinecone index")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))

    print("Ingestion completed")