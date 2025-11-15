from google import genai

def summarize_with_gemini(text: str) -> str:
    # Construct a clear prompt for Gemini
    prompt = (
        "This is text extracted from OCR technique. "
        "Summarize this and ONLY return the summarized text—no extra commentary or explanation."
        "\n\nText: " + text
    )

    # Initialize Gemini client (API key from env `GEMINI_API_KEY`)
    client = genai.Client()

    # Request summary from Gemini 2.5 Flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Return only the model's summarized output
    return response.text
