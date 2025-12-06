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

logger = logging.getLogger(__name__)


def process_document(file_path: str, doc_id: str) -> None:
    """
    Process a document through the full pipeline:
    1. Extract text with OCR
    2. Clean the text
    3. Correct OCR errors with LLM
    4. Chunk hierarchically
    5. Generate embeddings and store in vector DB
    6. Save parent chunks to MongoDB
    
    Args:
        file_path: Path to the document file
        doc_id: Unique document identifier for tracking
    
    Raises:
        Exception: If any step in the pipeline fails
    """
    try:
        logger.info(f"Starting document processing for doc_id: {doc_id}")
        
        # Step 1: Extract text from PDF
        logger.debug("Extracting text with OCR...")
        extracted_text = extract_text_with_ocr(file_path)
        
        # Step 2: Clean OCR text
        logger.debug("Cleaning OCR text...")
        cleaned_text = clean_ocr_text(extracted_text)
        
        # Update document status with processed text
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(update_document_status(
                doc_id, "processing", extracted_text, cleaned_text
            ))
        finally:
            loop.close()
        
        # Step 3: Correct text with LLM
        logger.debug("Correcting text with LLM...")
        corrected_text = correct_ocr_text(cleaned_text)
        
        # Step 4: Chunk text hierarchically
        logger.debug("Chunking text hierarchically...")
        parent_splitter, child_splitter = get_text_splitters()
        parent_chunks, child_chunks = chunk_text_hierarchically(
            corrected_text, parent_splitter, child_splitter
        )
        
        # Step 5: Setup vector database
        logger.debug("Setting up vector database...")
        index_name = "doc-analyzer-child-text"
        create_index_if_not_exists(index_name)
        embeddings = get_embedding_model()
        vector_store = get_vector_store(index_name, embeddings, create_if_not_exists=False)
        
        # Step 6: Add child chunks to vector DB
        logger.debug(f"Adding {len(child_chunks)} child chunks to vector store...")
        add_documents_to_vector_store(vector_store, child_chunks)
        
        # Step 7: Save parent chunks to MongoDB (async operation)
        logger.debug(f"Saving {len(parent_chunks)} parent chunks to MongoDB...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(save_parent_chunks(parent_chunks))
        finally:
            loop.close()
        
        # Update document status to completed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(update_document_status(doc_id, "completed"))
        finally:
            loop.close()
        
        logger.info(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        # Update document status to failed
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(update_document_status(doc_id, "failed"))
            finally:
                loop.close()
        except:
            pass
        raise RuntimeError("Error occur during processing the document..") from e

