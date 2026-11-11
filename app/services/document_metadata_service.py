"""Document metadata service using MongoDB for persistence.

Metadata is stored in MongoDB so it survives Render redeploys.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List
from app.config.mongodb_config import get_db

logger = logging.getLogger(__name__)
COLLECTION = "document_metadata"


async def save_document_metadata(doc_id: str, filename: str, filepath: str, status: str = "processing") -> None:
    db = await get_db()
    await db[COLLECTION].insert_one({
        "doc_id": doc_id,
        "filename": filename,
        "filepath": filepath,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed_text": None,
        "cleaned_text": None,
        "summary": None,
    })
    logger.info("Saved metadata: %s", doc_id)


async def update_document_status(doc_id: str, status: str, processed_text: Optional[str] = None, cleaned_text: Optional[str] = None) -> None:
    db = await get_db()
    update = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if processed_text is not None:
        update["processed_text"] = processed_text
    if cleaned_text is not None:
        update["cleaned_text"] = cleaned_text
    await db[COLLECTION].update_one({"doc_id": doc_id}, {"$set": update}, upsert=True)
    logger.info("Updated status: %s -> %s", doc_id, status)


async def save_document_summary(doc_id: str, summary: str) -> None:
    db = await get_db()
    await db[COLLECTION].update_one({"doc_id": doc_id}, {"$set": {"summary": summary, "updated_at": datetime.now(timezone.utc).isoformat()}})


async def get_document_metadata(doc_id: str) -> Optional[Dict]:
    db = await get_db()
    return await db[COLLECTION].find_one({"doc_id": doc_id}, {"_id": 0})


async def get_document_text(doc_id: str) -> Optional[str]:
    db = await get_db()
    doc = await db[COLLECTION].find_one({"doc_id": doc_id}, {"cleaned_text": 1, "processed_text": 1})
    if doc:
        return doc.get("cleaned_text") or doc.get("processed_text")
    return None


async def get_all_documents() -> List[Dict]:
    db = await get_db()
    cursor = db[COLLECTION].find({}, {"processed_text": 0, "cleaned_text": 0, "summary": 0, "_id": 0}).sort("created_at", -1).limit(50)
    return await cursor.to_list(length=50)
