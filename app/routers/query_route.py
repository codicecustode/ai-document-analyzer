"""Query and RAG routes with conversation memory and model selection."""
import asyncio
import uuid
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.query_service import format_query_for_search, get_query_embedding, build_rag_prompt
from app.services.retrieval_service import search_child_chunks
from app.services.mongodb_service import get_parent_chunks_by_ids
from app.services.llm_service import chat_llm_response, generate_llm_response, resolve_provider_model
from app.services.conversation_service import get_history, append_message, clear_history

logger = logging.getLogger(__name__)
router = APIRouter()

SYSTEM_PROMPT = (
    "You are a document analysis assistant. You answer questions about uploaded documents "
    "using the context retrieved from them. Be concise and accurate. If the context doesn't "
    "contain the answer, say so."
)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    query: str
    session_id: str
    provider: str
    model: str
    context_used: bool


@router.post("/query", response_model=QueryResponse, status_code=200)
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        provider, model = resolve_provider_model(request.provider, request.model)
        session_id = request.session_id or str(uuid.uuid4())

        # Step 1: Format query for vector search
        formatted_query = format_query_for_search(request.query)

        # Step 2: Generate embedding and search
        query_embedding = get_query_embedding(formatted_query)
        child_chunks = await asyncio.to_thread(search_child_chunks, query_embedding=query_embedding, top_k=request.top_k)

        context = ""
        if child_chunks:
            parent_ids = [c["parent_id"] for c in child_chunks if c.get("parent_id") is not None]
            if parent_ids:
                parent_chunks = await get_parent_chunks_by_ids(parent_ids)
                context = "".join(chunk.get("text", "") for chunk in parent_chunks)

        # Step 3: Build messages with conversation history
        history = await get_history(session_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if context:
            user_content = f"Context from documents:\n{context}\n\nQuestion: {request.query}"
        else:
            user_content = f"No relevant documents found. Question: {request.query}"

        messages.append({"role": "user", "content": user_content})

        # Step 4: Get LLM response with full conversation
        answer = chat_llm_response(messages, provider=provider, model=model)

        # Step 5: Save to conversation memory (store the clean query, not the context-stuffed one)
        await append_message(session_id, "user", request.query)
        await append_message(session_id, "assistant", answer)

        return QueryResponse(
            answer=answer,
            query=request.query,
            session_id=session_id,
            provider=provider,
            model=model,
            context_used=bool(context),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Query error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")


@router.delete("/query/history/{session_id}", status_code=200)
async def delete_history(session_id: str):
    await clear_history(session_id)
    return {"message": "Conversation history cleared", "session_id": session_id}
