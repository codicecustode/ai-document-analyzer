"""File upload service for handling document uploads."""
import os
from fastapi import UploadFile
from typing import Optional
import aiofiles


async def save_uploaded_file(file: UploadFile, upload_dir: Optional[str] = None) -> str:
    """
    Save an uploaded file to disk.
    
    Args:
        file: FastAPI UploadFile object
        upload_dir: Optional custom upload directory. If None, uses default 'app/uploads'
    
    Returns:
        str: Path to the saved file
    
    Raises:
        OSError: If file cannot be written
    """
    if upload_dir is None:
        # Get project root relative to this file
        current_file = os.path.abspath(__file__)
        app_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(app_dir)
        upload_dir = os.path.join(project_root, "uploads")
    
    os.makedirs(upload_dir, exist_ok=True)
    
    # Use UUID or timestamp to avoid filename conflicts
    import uuid
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    filepath = os.path.join(upload_dir, f"{file_id}{file_extension}")
    
    async with aiofiles.open(filepath, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # Read in 1MB chunks
            if not chunk:
                break
            await buffer.write(chunk)
    
    return filepath

