from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model(model_name="models/gemini-embedding-001"):
    return GoogleGenerativeAIEmbeddings(model=model_name)
