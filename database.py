import os
import logging
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# --- MongoDB Setup ---
def init_db_connection():
    """Initializes and returns the database collections."""
    logger.info("Loading environment variables for MongoDB")
    # Load environment variables from .env next to this file
    load_dotenv(Path(__file__).parent / ".env")

    db_password = os.getenv("MONGODB_PASSWORD")
    if not db_password:
        logger.error("MONGODB_PASSWORD not found in environment variables.")
        raise ValueError("MONGODB_PASSWORD environment variable is required.")

    username = quote_plus("syedbasitabbas")
    password = quote_plus(db_password)

    connection_string = f"mongodb+srv://{username}:{password}@userdata.114ih4x.mongodb.net/?retryWrites=true&w=majority&appName=UserData"

    mongo_client = None
    try:
        logger.info("Attempting to connect to MongoDB...")
        mongo_client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')

        # UserData Database and Collection
        db_userdata = mongo_client['UserData']
        user_collection = db_userdata['UserDataCollection']
        logger.info("Successfully connected to MongoDB and accessed UserData.UserDataCollection.")

        # Referrer_db Database and Collection
        db_referrer = mongo_client['referrer_db']
        referrers_match_collection = db_referrer['employee_embeddings']
        logger.info("Successfully accessed referrer_db.employee_embeddings.")
        
        return mongo_client, user_collection, referrers_match_collection

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        if mongo_client:
            mongo_client.close()
        raise

# Initialize connections on module load
MONGO_CLIENT, USER_COLLECTION, REFERRERS_MATCH_COLLECTION = init_db_connection()

def get_user_collection():
    return USER_COLLECTION

def get_referrers_collection():
    return REFERRERS_MATCH_COLLECTION

def close_db_connection():
    """Closes the MongoDB connection."""
    if MONGO_CLIENT:
        logger.info("Closing MongoDB connection...")
        MONGO_CLIENT.close()
        logger.info("MongoDB connection closed.") 