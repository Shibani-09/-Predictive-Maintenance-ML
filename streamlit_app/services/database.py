
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

# Locate the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load local environment variables from the root .env file.
load_dotenv(PROJECT_ROOT / ".env")


def _get_setting(name: str) -> str | None:
    """Read a setting from the environment or Streamlit secrets."""

    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        value = st.secrets.get(name)
    except FileNotFoundError:
        return None

    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


@st.cache_resource
def get_mongo_client():
    """Create and cache the MongoDB connection."""

    mongo_uri = _get_setting("MONGODB_URI")

    if not mongo_uri:
        raise ValueError(
            "MONGODB_URI must be configured in Streamlit secrets or the environment."
        )

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=15000
    )

    # Verify connection
    client.admin.command("ping")

    return client


def get_database():
    """Return the configured MongoDB database."""

    database_name = _get_setting("MONGODB_DATABASE")

    if not database_name:
        raise ValueError(
            "MONGODB_DATABASE must be configured in Streamlit secrets "
            "or the environment."
        )

    client = get_mongo_client()

    return client[database_name]


def get_users_collection():
    """Return the users collection."""

    return get_database()["users"]
