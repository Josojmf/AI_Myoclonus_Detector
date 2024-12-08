import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure, ConfigurationError
from dotenv import load_dotenv
import os
from flask import g


# Cargar variables de entorno
load_dotenv()

# Variables de configuración de MongoDB
db_username = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")
db_cluster = os.getenv("DB_CLUSTER")
db_name = os.getenv("DB_NAME")

logger = logging.getLogger('app_logger')


def connect_to_mongo():
    """
    Connect to MongoDB and return the client, database, and users collection.
    """
    try:
        # MongoDB connection details
        mongo_uri = (
            f"mongodb+srv://{db_username}:{db_password}@{db_cluster}/?retryWrites=true&w=majority&serverSelectionTimeoutMS=10000"
        )
        # Establish the connection
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        users_collection = db["Users"]

        # Test the connection
        client.admin.command("ping")
        logging.info("Connected to MongoDB successfully.")

        return client, db, users_collection

    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        return None, None, None
