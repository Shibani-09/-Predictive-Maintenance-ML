
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

# Locate the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from the root .env file
load_dotenv(PROJECT_ROOT / ".env")


@st.cache_resource
def get_mongo_client():
    """Create and cache the MongoDB connection."""

    mongo_uri = os.getenv("MONGODB_URI")

    if not mongo_uri:
        raise ValueError("MONGODB_URI is missing from the .env file.")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000
    )

    # Verify connection
    client.admin.command("ping")

    return client


def get_database():
    """Return the configured MongoDB database."""

    database_name = os.getenv("MONGODB_DATABASE")

    if not database_name:
        raise ValueError("MONGODB_DATABASE is missing from the .env file.")

    client = get_mongo_client()

    return client[database_name]


def get_users_collection():
    """Return the users collection."""

    return get_database()["users"]
