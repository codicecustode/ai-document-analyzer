from chunking_service import pc
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai


def generate_query_embedding(user_query: str):
    """
    Convert user query text to vector embedding.
    """
    try:
        embed = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return embed.embed_query(user_query)
    except Exception as e:
        raise RuntimeError(f"Error generating embedding: {e}")


def search_small_chunks(index, embedding, top_k=2):
    """
    Search smaller chunks first based on embedding similarity.
    """
    try:
        results = index.search(
            namespace="__default__",
            query={"inputs": {"text": f"{embedding}"}, "top_k": top_k},
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Error searching small chunks: {e}")


def search_large_chunks(index, parent_id: str, top_k=5):
    """
    Search full chunks based on previously matched small-chunk parent ID.
    """
    try:
        results = index.search(
            namespace="your-namespace",
            query={
                "inputs": {"text": "..."},
                "top_k": top_k,
                "filter": {"parent_id": parent_id},
            },
            fields=["parent_id", "page_content"],
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Error searching large chunks: {e}")



def build_prompt(user_query: str, context: str) -> str:
    """
    Build the prompt to send to the LLM.
    """
    return f"""
                You are a smart assistant helping users by answering their questions only using the provided context.

                User Query: {user_query}

                Context:
                {context}

                Instructions:
                - Only use information from the provided context when answering.
                - If the correct answer is NOT present in the context, respond with: "The answer is not found in the provided documents."
                - Do not hallucinate or make up information.
                - If reasoning is possible using the context, explain your reasoning.

                Final Answer:
            """


def llm_answer(prompt: str) -> str:
    """
    Send prompt to Gemini LLM and return its response.
    """
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error generating LLM answer: {e}")



def answer_user_query(user_query: str):
    """
    Full RAG pipeline:
    1. embed query
    2. search small chunks → get parent_id
    3. search large chunks using parent_id
    4. build prompt
    5. get final LLM answer
    """

    print("\n🚀 Running full RAG search pipeline...")

    # Embedding
    embedding = generate_query_embedding(user_query)

    # Search small chunks
    child_index = pc.Index(host="doc-analyzer-child-text")
    small_results = search_small_chunks(child_index, embedding)

    if not small_results.matches:
        return "No relevant document sections found."

    parent_id = small_results.matches[0].id  # top match

    # Search full chunks
    parent_index = pc.Index(host="doc-analyzer-parent-text")
    full_results = search_large_chunks(parent_index, parent_id)

    retrieved_docs = "\n".join(
        [doc["page_content"] for doc in full_results.matches]
    )

    # Step 4: Prompt build
    prompt = build_prompt(user_query, retrieved_docs)

    # Step 5: LLM answer
    final_answer = llm_answer(prompt)
    return final_answer


if __name__ == "__main__":
    query = "What does the agreement say about termination policy?"
    answer = answer_user_query(query)
    print("\nFinal Answer:\n", answer)
