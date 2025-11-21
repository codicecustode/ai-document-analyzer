from fastapi import APIRouter
from services.ocr_service import extract_text_with_ocr
from services.cleaning_service import  clean_ocr_for_chunking
from services.vector_db_service import initialize_pinecone, create_index_if_not_exists
from services.chunking_service import get_splitters, chunk_text_hierarchical
from services.vector_db_indexing_service import setup_vector_stores, add_documents
from services.embedding_service import embedding_model


router = APIRouter()

@router.post('/chat')
async def start_chat(file_path: str):

  #get the text from pdf
  extracted_text = extract_text_with_ocr(file_path)

  #clean the ocr text
  cleaned_text = clean_ocr_for_chunking(extracted_text)

  #get the splitter for parent and child
  parent_splitter, child_splitter = get_splitters()

  #chunk doc into child and parent for small to big retrival
  parent_chunk_doc, child_chunk_doc = chunk_text_hierarchical(cleaned_text, parent_splitter, child_splitter )

  #initialize pinecone db for storing the vectors
  pc = initialize_pinecone()

  #create index if not exist
  create_index_if_not_exists(pc, "doc-analyzer-parent-text")
  create_index_if_not_exists(pc, "doc-analyzer-child-text")
  
  #parent_store child_store
  parent_store, child_store = setup_vector_stores(pc, "doc-analyzer-parent-text", "doc-analyzer-child-text", embedding_model)

  #add document in vector database  
  add_documents(parent_store, child_store, parent_chunk_doc, child_chunk_doc)
  