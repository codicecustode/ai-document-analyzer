from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

def setup_vector_stores(pc, parent_index_name, child_index_name, embeddings):

    parent_index = pc.Index(parent_index_name)
    child_index = pc.Index(child_index_name)

    if not pc.has_index(parent_index):
      pc.create_index(
          name=parent_index_name,
          dimension=3072,
          metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
      )
    if not pc.has_index(child_index):
      pc.create_index(
          name=child_index_name,
          dimension=3072,
          metric="cosine",
          spec=ServerlessSpec(cloud="aws", region="us-east-1"),
      )

    parent_store = PineconeVectorStore(index=parent_index, embedding=embeddings)
    child_store = PineconeVectorStore(index=child_index, embedding=embeddings)

    return parent_store, child_store


def add_documents(parent_store, child_store, parent_docs, child_docs):
    parent_store.add_documents(parent_docs)
    child_store.add_documents(child_docs)
