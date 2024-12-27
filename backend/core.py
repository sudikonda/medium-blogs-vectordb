import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    doc_search = PineconeVectorStore(
        index_name="langchain-doc-index",
        embedding=embeddings
    )
    chat = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(retriever=doc_search.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrival_chain.invoke(input={"input": query})
    return result


if __name__ == '__main__':
    result = run_llm("What is a Langchain Generative Agent?")
    print(result["answer"])