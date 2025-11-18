import re
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def clean_ocr_for_chunking(ocr_text: str) -> str:
    """
    Cleans OCR-extracted text by repairing hyphenations, merging lines,
    removing noise, and normalizing text.
    """
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', ocr_text)
    text = re.sub(r'(?<![.\?!])\n(?!\n)', ' ', text)
    text = re.sub(r'^\s*[\dIVXLCDMivxlcdm]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('“', '"').replace('”', '"').replace("’", "'")
    return text

def get_splitters(parent_size=1500, parent_overlap=200,
                  child_size=500, child_overlap=100):
    """
    Returns configured RecursiveCharacterTextSplitter instances for
    parent and child chunking.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    return parent_splitter, child_splitter

def chunk_text_hierarchical(document_text: str, parent_splitter, child_splitter):
    """
    Splits text into hierarchical chunks: parent chunks and child chunks with metadata.
    Returns two lists of Document objects: parent_documents and child_documents.
    """
    parent_chunks = parent_splitter.split_text(document_text)
    child_documents = []
    for idx, parent_chunk in enumerate(parent_chunks):
        child_chunks = child_splitter.split_text(parent_chunk)
        for child_text in child_chunks:
            child_documents.append(Document(
                page_content=child_text,
                metadata={"parent_id": idx}
            ))
    parent_documents = [Document(page_content=chunk, metadata={"parent_id": idx})
                        for idx, chunk in enumerate(parent_chunks)]
    return parent_documents, child_documents

def initialize_pinecone(api_key: str):
    """
    Initialize Pinecone client or raise error if API key missing.
    """
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from environment variables")
    return Pinecone(api_key=api_key)

def create_pinecone_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int = 1536):
    """
    Create a Pinecone index if it does not exist, with default serverless spec.
    """
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def setup_vector_stores(parent_index_name: str, child_index_name: str, embeddings_model: str):
    """
    Create OpenAI embeddings and initialize PineconeVectorStore instances for both parent and child indexes.
    """
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    parent_vector_store = PineconeVectorStore(index=parent_index_name, embedding=embeddings)
    child_vector_store = PineconeVectorStore(index=child_index_name, embedding=embeddings)
    return parent_vector_store, child_vector_store

def index_documents(parent_vector_store, child_vector_store, parent_documents, child_documents):
    """
    Add documents to the respective Pinecone vector stores.
    """
    parent_vector_store.add_documents(documents=parent_documents)
    child_vector_store.add_documents(documents=child_documents)

def main(document_text: str):
    """
    Orchestrates the entire pipeline from cleaning text, chunking, and indexing.
    """
    cleaned_text = clean_ocr_for_chunking(document_text)
    parent_splitter, child_splitter = get_splitters()
    parent_docs, child_docs = chunk_text_hierarchical(cleaned_text, parent_splitter, child_splitter)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = initialize_pinecone(pinecone_api_key)

    parent_index_name = "doc_analyzer_parent_text"
    child_index_name = "doc_analyzer_child_text"

    create_pinecone_index_if_not_exists(pc, parent_index_name)
    create_pinecone_index_if_not_exists(pc, child_index_name)

    parent_vector_store, child_vector_store = setup_vector_stores(
        parent_index_name, child_index_name, embeddings_model="text-embedding-3-small"
    )

    index_documents(parent_vector_store, child_vector_store, parent_docs, child_docs)
    print("Document indexing complete.")

