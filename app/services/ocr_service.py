from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

def extract_text_with_ocr(file_path):
  try:
    images = convert_from_path(file_path)
  except PDFInfoNotInstalledError:
    print("Poppler utils not installed or not found.")
  except PDFPageCountError:
    print("Could not get PDF page count.")
  except PDFSyntaxError:
    print("PDF file is corrupted or has syntax errors.")
    
  doc_text = ""
  
  for image in images:
    doc_text += pytesseract.image_to_string(Image.open(image))
  return doc_text