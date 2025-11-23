from fastapi import APIRouter
from services.query_answer_service import query_embedding, search_small_chunks, search_large_chunks, format_user_query, build_prompt, answer_user_query

router = APIRouter()


@router.post("/ask")
async def ask_query(user_query: str):
  
  formatted_user_query = format_user_query(user_query)
  user_query_embedding = query_embedding(formatted_user_query)
  small_chunk_result = search_small_chunks("doc-analyzer-child-text", user_query_embedding)
  query_answer_context = search_large_chunks("doc-analyzer-parent-text", small_chunk_result)
  llm_prompt = build_prompt(query_answer_context)
  query_answer = answer_user_query(llm_prompt)
  return query_answer