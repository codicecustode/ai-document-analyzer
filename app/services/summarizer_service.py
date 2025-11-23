from google import genai
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file into environment

def summarize_with_gemini(text: str) -> str:
    # Construct a clear prompt for Gemini
    prompt = (
      "The following text is extracted via OCR and may contain spelling or grammar errors."
      "Carefully correct obvious errors but do not add or invent new information."
      "Summarize all key points concisely, using bullet points or short sections."
      "Omit any introductory text, explanations, or 'Here is a summary...' commentary."
      "Retain the original meaning and structure where possible, but improve readability and clarity."
      "If the document has sections (offers, terms, names, dates, policies, conditions, etc.), separate them with headings or clear bullets."
      "Only output the cleaned, summarized document content."
      "\n\nText:\n" + text
    )


    try:
    # Initialize Gemini client (API key from env `GEMINI_API_KEY`)
      client = genai.Client()

      # Request summary from Gemini 2.5 Flash
      response = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt
      )

      # Return only the model's summarized output
      return response.text
    except Exception as e:
      print("Google Gemini Error-->",e)

#Example usage:
#summary = summarize_with_gemini("dont summarize tell me ur version")
#print(summary)
