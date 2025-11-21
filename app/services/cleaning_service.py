import re

def clean_ocr_for_chunking(ocr_text: str) -> str:
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', ocr_text)
    text = re.sub(r'(?<![.\?!])\n(?!\n)', ' ', text)
    text = re.sub(r'^\s*[\dIVXLCDMivxlcdm]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.replace('“', '"').replace('”', '"').replace("’", "'")
