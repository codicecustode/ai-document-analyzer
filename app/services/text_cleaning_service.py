"""Text cleaning service for preprocessing OCR text."""
import re


def clean_ocr_text(ocr_text: str) -> str:
    """
    Clean OCR-extracted text for better processing.
    
    Removes:
    - Hyphenated words split across lines
    - Unnecessary line breaks
    - Standalone roman numerals
    - Extra whitespace
    - Smart quotes
    
    Args:
        ocr_text: Raw text from OCR extraction
    
    Returns:
        str: Cleaned text ready for further processing
    """
    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', ocr_text)
    
    # Remove line breaks that aren't paragraph breaks
    text = re.sub(r'(?<![.\?!])\n(?!\n)', ' ', text)
    
    # Remove standalone roman numerals
    text = re.sub(r'^\s*[\dIVXLCDMivxlcdm]+\s*$', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Replace smart quotes with standard quotes
    text = text.replace('"', '"').replace('"', '"').replace("'", "'")
    
    return text

