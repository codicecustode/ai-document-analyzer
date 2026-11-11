"""Document processing pipeline orchestrator."""
import asyncio
import logging
import time
from app.services.document_extraction_service import extract_text_with_ocr
from app.services.text_cleaning_service import clean_ocr_text
from app.services.chunking_service import get_text_splitters, chunk_text_hierarchically
from app.services.embedding_service import get_embedding_model
from app.services.vector_db_service import (
    get_vector_store,
    add_documents_to_vector_store,
    create_index_if_not_exists,
)
from app.services.mongodb_service import save_parent_chunks
from app.services.document_metadata_service import update_document_status

logger = logging.getLogger(__name__)


async def process_document(file_path: str, doc_id: str) -> None:
    start = time.time()
    try:
        logger.info("Processing %s", doc_id)

        # Step 1: Extract and clean text
        extracted_text = await asyncio.to_thread(extract_text_with_ocr, file_path)
        cleaned_text = await asyncio.to_thread(clean_ocr_text, extracted_text)
        logger.info("Extracted %d chars in %.1fs", len(cleaned_text), time.time() - start)

        await update_document_status(doc_id, "processing", extracted_text, cleaned_text)

        # Step 2: Chunk (skip LLM correction - it slows down the pipeline and often fails)
        parent_chunks, child_chunks = await asyncio.to_thread(
            chunk_text_hierarchically, cleaned_text, *get_text_splitters()
        )
        logger.info("Chunked into %d parent, %d child in %.1fs", len(parent_chunks), len(child_chunks), time.time() - start)

        # Step 3: Setup vector DB
        index_name = "doc-analyzer-child-text"
        await asyncio.to_thread(create_index_if_not_exists, index_name)
        embeddings = await asyncio.to_thread(get_embedding_model)
        vector_store = await asyncio.to_thread(
            get_vector_store, index_name, embeddings, create_if_not_exists=False
        )

        # Step 4: Upload embeddings + save chunks in parallel
        await asyncio.gather(
            asyncio.to_thread(add_documents_to_vector_store, vector_store, child_chunks),
            save_parent_chunks(parent_chunks),
        )

        await update_document_status(doc_id, "completed")
        logger.info("Document %s completed in %.1fs", doc_id, time.time() - start)

    except Exception as e:
        logger.error("Processing failed for %s: %s", doc_id, e, exc_info=True)
        await update_document_status(doc_id, "failed")
        raise
