from cachetools import TTLCache
from pymongo.collection import Collection
from models import SessionState
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# A thread-safe cache to hold active session objects in memory for 1 hour.
# This reduces database lookups for active users.
SESSION_CACHE = TTLCache(maxsize=1000, ttl=3600)

def get_session(session_id: str, db_collection: Collection) -> SessionState:
    """
    Retrieves a session state. It checks the in-memory cache first, then falls back
    to MongoDB. If not found in either, it creates a new session object.

    Args:
        session_id: The unique identifier for the session.
        db_collection: The MongoDB collection where sessions are stored.

    Returns:
        The SessionState object for the given session_id.
    """
    # 1. Check in-memory cache first for performance
    session = SESSION_CACHE.get(session_id)
    if session:
        logger.debug(f"Session '{session_id}' found in cache.")
        return session

    # 2. If not in cache, try to load from MongoDB
    logger.debug(f"Session '{session_id}' not in cache, checking MongoDB.")
    db_data = db_collection.find_one({"session_id": session_id})
    if db_data:
        try:
            session_state = SessionState(**db_data)
            SESSION_CACHE[session_id] = session_state  # Load into cache
            logger.info(f"Session '{session_id}' loaded from MongoDB into cache.")
            return session_state
        except Exception as e:
            # If data in DB is corrupted or doesn't match the model, log error and create a fresh session
            logger.error(f"Error loading session '{session_id}' from DB due to data mismatch: {e}. A new session will be created.")
            # Fall through to create a new session

    # 3. If not found anywhere, create a new session
    logger.info(f"No existing session found for '{session_id}'. Creating a new one.")
    new_session = SessionState(session_id=session_id)
    SESSION_CACHE[session_id] = new_session
    return new_session

def save_session(session_state: SessionState, db_collection: Collection):
    """
    Saves the session state to both the in-memory cache and MongoDB.

    Args:
        session_state: The SessionState object to save.
        db_collection: The MongoDB collection for persistence.
    """
    session_id = session_state.session_id
    session_state.update_timestamp() # Ensure last_updated is fresh

    # 1. Update the cache
    SESSION_CACHE[session_id] = session_state
    logger.debug(f"Session '{session_id}' updated in cache.")

    # 2. Persist to MongoDB
    try:
        # Convert the Pydantic model to a dictionary for MongoDB insertion/update
        session_dict = session_state.dict(by_alias=True)
        db_collection.update_one(
            {"session_id": session_id},
            {"$set": session_dict},
            upsert=True
        )
        logger.debug(f"Session '{session_id}' saved to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to save session '{session_id}' to MongoDB: {e}") 