from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader(
        "/Users/sudikonda/Developer/Python/medium-blogs-vectordb/langchain-docs"
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")

    for chunk in chunks:
        new_url = chunk.metadata.get("source")
        new_url = new_url.replace("/Users/sudikonda/Developer/Python/medium-blogs-vectordb/langchain-docs", "https:/")
        chunk.metadata["source"] = new_url

    PineconeVectorStore.from_documents(chunks, embeddings, index_name="langchain-doc-index")

    print("Ingestion completed")


if __name__ == '__main__':
    ingest_docs()
