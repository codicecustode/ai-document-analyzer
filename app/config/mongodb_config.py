import os

import asyncio
from pymongo import AsyncMongoClient, ASCENDING

DB_INSTANCE = None
client = None
async def initialize_mongodb():
  global client, DB_INSTANCE
  MONGO_DB_URI = os.getenv("MONGO_DB_URI")
  DB_NAME = os.getenv("MONGO_DB_NAME")
  client = AsyncMongoClient(MONGO_DB_URI)
  try:
    database = client.get_database(DB_NAME)
    DB_INSTANCE = database
    # Create index on 'parent_id' field in 'parentchunk_collection'
    await database.parent_doc_chunk_collection.create_index([("parent_id", ASCENDING)], background=True)
  except Exception as e:
    raise Exception("Unable to find the document due to the following error: ", e)


async def get_db():
  global DB_INSTANCE
  if DB_INSTANCE == None:
    await initialize_mongodb()
  return DB_INSTANCE
  