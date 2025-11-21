from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_splitters(parent_size=1500, parent_overlap=200,
                  child_size=500, child_overlap=100):

    parent = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
    )
    child = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
    )
    return parent, child


def chunk_text_hierarchical(document_text, parent_splitter, child_splitter):
    parent_chunks = parent_splitter.split_text(document_text)
    child_docs = []

    for idx, p_chunk in enumerate(parent_chunks):
        for c_chunk in child_splitter.split_text(p_chunk):
            child_docs.append(Document(
                page_content=c_chunk,
                metadata={"parent_id": idx}
            ))

    parent_docs = [
        Document(page_content=p, metadata={"parent_id": i})
        for i, p in enumerate(parent_chunks)
    ]

    return parent_docs, child_docs
