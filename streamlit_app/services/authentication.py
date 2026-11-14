
from datetime import datetime, timezone

from passlib.context import CryptContext
from pymongo.errors import DuplicateKeyError

from streamlit_app.services.database import get_users_collection


# Password hashing configuration
password_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password):
    """Hash a password using bcrypt."""
    return password_context.hash(password)


def verify_password(password, password_hash):
    """Verify a password against its bcrypt hash."""
    return password_context.verify(password, password_hash)


def register_user(full_name, email, password):
    """Register a new user in MongoDB."""

    collection = get_users_collection()

    full_name = full_name.strip()
    email = email.strip().lower()

    if not full_name or not email or not password:
        return False, "All fields are required."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    existing_user = collection.find_one({
        "normalized_email": email
    })

    if existing_user:
        return False, "An account with this email already exists."

    user = {
        "full_name": full_name,
        "email": email,
        "normalized_email": email,
        "password_hash": hash_password(password),
        "failed_login_attempts": 0,
        "is_locked": False,
        "created_at": datetime.now(timezone.utc)
    }

    try:
        collection.insert_one(user)
        return True, "Account created successfully."

    except DuplicateKeyError as error:
        key_pattern = (error.details or {}).get("keyPattern")
        if key_pattern == {"normalized_email": 1}:
            return False, "An account with this email already exists."
        raise


def authenticate_user(email, password):
    """Authenticate an existing user."""

    collection = get_users_collection()
    email = email.strip().lower()

    user = collection.find_one({
        "normalized_email": email
    })

    if not user:
        return False, "Invalid email or password.", None

    if user.get("is_locked", False):
        return False, "Your account is locked. Contact the administrator.", None

    if not verify_password(password, user["password_hash"]):
        attempts = user.get("failed_login_attempts", 0) + 1

        collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "failed_login_attempts": attempts,
                    "is_locked": attempts >= 3
                }
            }
        )

        if attempts >= 3:
            return False, "Account locked after 3 failed attempts.", None

        return False, f"Invalid password. {3 - attempts} attempts remaining.", None

    # Reset failed attempts after successful login
    collection.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "failed_login_attempts": 0
            }
        }
    )

    return True, "Login successful.", {
        "full_name": user.get("full_name", ""),
        "email": user.get("email", "")
    }
