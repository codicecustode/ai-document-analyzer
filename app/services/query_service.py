"""Query processing service for user queries."""
import logging
from typing import List
from services.llm_service import generate_llm_response
from services.embedding_service import get_embedding_model

logger = logging.getLogger(__name__)


def format_query_for_search(user_query: str) -> str:
    """
    Format and correct user query for better vector search results.
    
    Args:
        user_query: Raw user query
    
    Returns:
        str: Formatted and corrected query
    """
    prompt = (
        "Correct only spelling and grammatical mistakes in the following user query for vector search. "
        "Return ONLY the fixed query, without changing the query's meaning, intent, or adding any extra details. "
        "Do NOT include explanations, instructions, or additional tokens—simply output the corrected user query ONLY.\n"
        f"User query: {user_query}"
    )
    
    try:
        formatted_query = generate_llm_response(prompt, model="gemini-2.5-pro")
        return formatted_query.strip()
    except Exception as e:
        logger.warning(f"Query formatting failed, using original query: {e}")
        return user_query


def get_query_embedding(query: str) -> List[float]:
    """
    Convert query text to vector embedding.
    
    Args:
        query: Query text
    
    Returns:
        List of floats representing the embedding vector
    """
    try:
        embedding_model = get_embedding_model()
        return embedding_model.embed_query(query)
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}") from e


def build_rag_prompt(user_query: str, context: str) -> str:
    """
    Build prompt for RAG (Retrieval-Augmented Generation).
    
    Args:
        user_query: User's question
        context: Retrieved context from documents
    
    Returns:
        str: Formatted prompt for LLM
    """
    return f"""You are an assistant that answers the user's question strictly using the provided context.

                User Query:
                {user_query}

                Context:
                {context}

                Rules:
                1. Use only information found in the context.
                2. Do not add, assume, or infer anything that is not explicitly in the context.
                3. The final answer must be written clearly for the end user.
                4. Do not include any explanations about rules, context, or reasoning in the final answer.
                5. If the answer is not present in the context, reply exactly with:
                "The answer is not found in the provided documents."

                Final Answer:
            """

