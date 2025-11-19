import re
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
    pc = Pinecone(api_key=api_key)
    return pc

pc = initialize_pinecone()

def create_pinecone_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int = 3072):
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

def setup_vector_stores(pc: Pinecone, parent_index_name: str, child_index_name: str, embeddings_model: str):
    """
    Create Google embeddings and initialize PineconeVectorStore instances for both parent and child indexes.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)    
    parent_index_name=pc.Index(parent_index_name)
    child_index_name=pc.Index(child_index_name)
    
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
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vec = emb.embed_query("hello world")
    print("ACTUAL EMBEDDING DIM:", len(vec))
    """
    Orchestrates the entire pipeline from cleaning text, chunking, and indexing.
    """
    cleaned_text = clean_ocr_for_chunking(document_text)
    parent_splitter, child_splitter = get_splitters()
    parent_docs, child_docs = chunk_text_hierarchical(cleaned_text, parent_splitter, child_splitter)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    parent_index_name = "doc-analyzer-parent-text"
    child_index_name = "doc-analyzer-child-text"

    create_pinecone_index_if_not_exists(pc, parent_index_name)
    create_pinecone_index_if_not_exists(pc, child_index_name)

    parent_vector_store, child_vector_store = setup_vector_stores(
        pc, parent_index_name, child_index_name, embeddings_model="models/gemini-embedding-001"
    )

    index_documents(parent_vector_store, child_vector_store, parent_docs, child_docs)
    print("Document indexing complete.")

# Usage example:
document_text = """
In the realm of modern digital operations, organizations face a rapidly evolving landscape of challenges, opportunities, and responsibilities. Over the past decade, digital transformation has shifted from being a competitive advantage to a fundamental necessity. Enterprises, small businesses, and public institutions now rely heavily on data-driven decision-making, automated systems, and real-time analytics to achieve efficiency, maintain security, and deliver value to their stakeholders.

One of the most critical components of this shift is the ability to collect, process, and interpret unstructured data. Documents such as contracts, invoices, research papers, medical reports, legal notices, handwritten notes, and historical archives contain immense value. However, much of this information is often locked behind inconsistent formatting, poor scan quality, OCR errors, and ambiguous structures. To unlock this information, organizations must employ advanced text extraction and intelligent retrieval systems that can accurately interpret context, maintain structural meaning, and provide reliable outputs.

Over the years, advancements in Artificial Intelligence—especially in Natural Language Processing (NLP)—have made it possible to analyze extremely large datasets with high precision. Models trained on billions of parameters, combined with modern vector databases, allow applications to understand semantic meaning instead of relying on simple keyword matching. This shift marks a significant leap from traditional search systems, enabling intelligent assistants, legal analyzers, customer support bots, research summarizers, and document compliance tools.

Despite this progress, several challenges persist. For example, scanned PDFs may contain noise, misaligned characters, random page numbers, or broken words such as "docu-
ment" or "infor-
mation," which corrupt the extracted text. Some documents span multiple sections, chapters, or structures—yet OCR engines flatten everything into raw text. This creates difficulties when models attempt to understand hierarchy, relationships, or logical connections.

To solve these problems, modern document processing systems use multi-layer chunking strategies. A common technique involves splitting data into parent chunks (large, meaningful sections) and child chunks (smaller, fine-grained segments). The parent chunks preserve full context, while child chunks support precise retrieval. When a query is asked, the system retrieves the most relevant child chunk, identifies its parent chunk through metadata, and provides a comprehensive answer. This small-to-big retrieval strategy greatly improves accuracy by combining local relevance with global understanding.

Additionally, vector databases such as Pinecone, Weaviate, and Milvus provide highly scalable environments for storing and searching dense embeddings. Instead of storing words or paragraphs, these systems store numerical vector representations generated by embedding models like `text-embedding-3-small`. These vectors encode semantics, enabling the system to detect conceptual similarities—for example, understanding that "contract termination clause" is related to "legal agreement cancellation terms" even if the words do not match exactly.

As digital ecosystems continue expanding, organizations increasingly require automated pipelines capable of ingesting thousands of documents, cleaning them, chunking them intelligently, embedding the chunks, and storing them in vector databases. With this infrastructure, they can build advanced AI-driven applications that summarize contracts, detect compliance issues, classify medical reports, answer questions from research journals, extract insights from financial statements, and support large-scale enterprise knowledge management.

The future of document intelligence lies in combining high-quality OCR correction, hierarchical chunking, powerful embeddings, and efficient vector search. Together, these components form the foundation of next-generation Retrieval-Augmented Generation (RAG) systems that make information accessible, searchable, and actionable at unprecedented scale.
"""

main(document_text)
