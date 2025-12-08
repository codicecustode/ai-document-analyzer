"""Query and RAG (Retrieval-Augmented Generation) routes."""

from asyncio import to_thread
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.query_service import (
    format_query_for_search,
    get_query_embedding,
    build_rag_prompt,
)
from app.services.retrieval_service import search_child_chunks
from app.services.mongodb_service import get_parent_chunks_by_ids
from app.services.llm_service import generate_llm_response

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query."""

    query: str
    top_k: Optional[int] = 3

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the salary mentioned in the offer letter?",
                "top_k": 3,
            }
        }


class QueryResponse(BaseModel):
    """Response model for query answer."""

    answer: str
    query: str
    context_used: bool

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The salary mentioned in the offer letter is $100,000 per year.",
                "query": "What is the salary mentioned in the offer letter?",
                "context_used": True,
            }
        }


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=200,
    summary="Query documents",
    description="Ask a question about the uploaded documents using RAG (Retrieval-Augmented Generation).",
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query documents using RAG (Retrieval-Augmented Generation).

    This endpoint:
    1. Formats and corrects the user query
    2. Converts query to embedding vector
    3. Searches for relevant document chunks
    4. Retrieves full context from parent chunks
    5. Generates an answer using LLM with the retrieved context
    """
    try:
        logger.info(f"Processing query: {request.query}")

        # Step 1: Format and correct the query
        formatted_query = format_query_for_search(request.query)
        logger.debug(f"Formatted query: {formatted_query}")

        # Step 2: Generate query embedding
        query_embedding = get_query_embedding(formatted_query)
        logger.debug("Query embedding generated")

        # Step 3: Search for similar child chunks
        child_chunks = await asyncio.to_thread(
            search_child_chunks,query_embedding=query_embedding, top_k=request.top_k
        )
        logger.debug(f"Found {len(child_chunks)} relevant child chunks")

        if not child_chunks:
            return QueryResponse(
                answer="No relevant information found in the documents.",
                query=request.query,
                context_used=False,
            )

        # Step 4: Get parent chunks for full context
        parent_ids = [
            chunk["parent_id"]
            for chunk in child_chunks
            if chunk.get("parent_id") is not None
        ]

        if not parent_ids:
            return QueryResponse(
                answer="Unable to retrieve context from documents.",
                query=request.query,
                context_used=False,
            )

        parent_chunks = await get_parent_chunks_by_ids(parent_ids)

        # Concatenate parent chunk texts for context
        context = "".join(chunk.get("text", "") for chunk in parent_chunks)

        if not context:
            return QueryResponse(
                answer="No context available to answer the query.",
                query=request.query,
                context_used=False,
            )

        logger.debug(f"Retrieved context length: {len(context)} characters")

        # Step 5: Build RAG prompt
        rag_prompt = build_rag_prompt(request.query, context)

        # Step 6: Generate answer using LLM
        answer = generate_llm_response(rag_prompt, model="gemini-2.5-flash")

        logger.info(f"Query answered successfully for: {request.query}")

        return QueryResponse(answer=answer, query=request.query, context_used=True)

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process query: {str(e)}"
        )
