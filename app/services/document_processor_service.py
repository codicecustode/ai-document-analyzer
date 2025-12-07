"""Document processing pipeline orchestrator."""
import asyncio
import logging
from services.document_extraction_service import extract_text_with_ocr
from services.text_cleaning_service import clean_ocr_text
from services.text_correction_service import correct_ocr_text
from services.chunking_service import get_text_splitters, chunk_text_hierarchically
from services.embedding_service import get_embedding_model
from services.vector_db_service import (
    get_vector_store,
    add_documents_to_vector_store,
    create_index_if_not_exists
)
from services.mongodb_service import save_parent_chunks
from services.document_metadata_service import update_document_status
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# Global thread pool for CPU-bound work
executor = ThreadPoolExecutor(max_workers=4)

async def process_document(file_path: str, doc_id: str) -> None:
    try:
        logger.info(f"Starting document processing for doc_id: {doc_id}")
        
        # ✅ Offload ALL sync blocking steps to threadpool
        extracted_text = await asyncio.to_thread(extract_text_with_ocr, file_path)
        cleaned_text = await asyncio.to_thread(clean_ocr_text, extracted_text)
        
        await update_document_status(doc_id, "processing", extracted_text, cleaned_text)
        
        corrected_text = await asyncio.to_thread(correct_ocr_text, cleaned_text)
        parent_chunks, child_chunks = await asyncio.to_thread(
            chunk_text_hierarchically, corrected_text, 
            *get_text_splitters()  # Assuming this returns splitters
        )
        
        # Vector DB setup (if sync, offload)
        index_name = "doc-analyzer-child-text"
        await asyncio.to_thread(create_index_if_not_exists, index_name)
        embeddings = await asyncio.to_thread(get_embedding_model)
        vector_store = await asyncio.to_thread(
            get_vector_store, index_name, embeddings, create_if_not_exists=False
        )
        
        # ✅ Now gather truly runs in parallel
        task_vector = asyncio.to_thread(add_documents_to_vector_store, vector_store, child_chunks)
        task_mongo =  save_parent_chunks(parent_chunks)  # Already async
        
        await asyncio.gather(task_vector, task_mongo)
        await update_document_status(doc_id, "completed")
        
    except Exception as e:
        await update_document_status(doc_id, "failed")
        raise