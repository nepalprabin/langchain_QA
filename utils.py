import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Creating embeddings for semantic search
import pinecone

# loading the data


def load_data(file):
    loader = UnstructuredPDFLoader(file)
    data = loader.load()
    return data

# chunking


def convert_into_chunks(chunk_size, data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts


# Chunking data upto smaller components
def embeddings_(openai_api_key):
    return OpenAIEmbeddings(openai_api_key=openai_api_key)

# initializing pinecone

# create pinecone index


def create_index(index_name):
    pinecone.create_index(index_name, dimension=1536,
                          metric="cosine", pods=1, pod_type="p1.x1")


def delete_index(index_name):
    pinecone.delete_index(index_name)


def vector_database_setup(api_key, environment):
    pinecone.init(
        api_key=api_key,
        environment=environment
    )


def store_to_pinecone(texts, embeddings, index_name):
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name)
    return docsearch


def docs_(docsearch, query):
    return docsearch.similarity_search(query)
