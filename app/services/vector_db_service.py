import os
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY missing")
    return Pinecone(api_key=api_key)


def create_index_if_not_exists(pc, name: str, dimension=3072):
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
