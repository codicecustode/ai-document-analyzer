"""Conversation memory stored in MongoDB.

Each conversation is a list of messages (role + content) tied to a session_id.
When the user switches models mid-conversation, the history stays because it's
stored in the database, not in the LLM client.
"""
import logging
from datetime import datetime, timezone
from typing import Optional
from app.config.mongodb_config import get_db

logger = logging.getLogger(__name__)

COLLECTION = "conversations"
MAX_HISTORY = 20  # keep the last N turns to stay within context limits


async def get_history(session_id: str) -> list[dict]:
    """Get the conversation history for a session."""
    db = await get_db()
    doc = await db[COLLECTION].find_one({"session_id": session_id})
    if not doc:
        return []
    return doc.get("messages", [])[-MAX_HISTORY:]


async def append_message(session_id: str, role: str, content: str) -> None:
    """Append a message to a conversation, creating it if needed."""
    db = await get_db()
    message = {"role": role, "content": content, "ts": datetime.now(timezone.utc).isoformat()}
    await db[COLLECTION].update_one(
        {"session_id": session_id},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": datetime.now(timezone.utc).isoformat()},
            "$setOnInsert": {"created_at": datetime.now(timezone.utc).isoformat()},
        },
        upsert=True,
    )


async def clear_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    db = await get_db()
    await db[COLLECTION].delete_one({"session_id": session_id})


async def list_sessions(limit: int = 20) -> list[dict]:
    """List recent conversation sessions."""
    db = await get_db()
    cursor = db[COLLECTION].find(
        {}, {"session_id": 1, "created_at": 1, "updated_at": 1, "_id": 0}
    ).sort("updated_at", -1).limit(limit)
    return await cursor.to_list(length=limit)
