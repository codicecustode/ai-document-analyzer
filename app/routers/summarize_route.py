from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile
import os
from services.ocr_service import extract_text_with_ocr
from services.summarizer_service import summarize_with_gemini
from services.cleaning_service import clean_ocr_for_chunking

router = APIRouter()

@router.post('/upload_file')
async def create_upload_file(file: UploadFile = File(...)):
  try:
    # mode here is for permission
    THIS_FILE = os.path.abspath(__file__)
    APP_DIR = os.path.dirname(THIS_FILE)
    PROJECT_ROOT = os.path.dirname(APP_DIR)
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as buffer:
      while True:
          chunk = await file.read(1024 * 1024)
          if not chunk:
              break
          buffer.write(chunk)
    
    #get the text from ocr
    pdf_text = extract_text_with_ocr(filepath)
    print("pdf_text", pdf_text)
    cleaned_text = clean_ocr_for_chunking(pdf_text)
    #get the summarized vesrsion
    summarized_text = summarize_with_gemini(cleaned_text)
    print("summarized_text", summarized_text)
    #return summarized version
    return summarized_text
    
  except Exception as e:
    print(f"Getting Error while summarizing the text using gemini: {e}")
    
