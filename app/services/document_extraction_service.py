"""Document extraction service for OCR and text extraction from PDFs."""
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import logging

logger = logging.getLogger(__name__)


def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF using OCR (Optical Character Recognition).
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        str: Extracted text from all pages
    
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If OCR extraction fails
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image)
            full_text += text + "\n"
        
        doc.close()
        return full_text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise

