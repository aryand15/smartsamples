"""
Data models for SmartSamples application.

This module defines the SQLModel classes that represent the database schema
for users, samples, downloads, and upvotes. These models provide data
validation and ORM functionality for PostgreSQL database interactions.
"""
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone


class User(SQLModel, table=True):
    """
    User model representing registered users in the system.

    Attributes:
        id: Primary key, auto-generated
        username: Unique username for login
        password_hash: Bcrypt hashed password
        bio: Optional user biography
        email: Unique email address
        created_at: Timestamp of account creation
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str = Field()
    bio: Optional[str] = Field(default="")
    email: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Sample(SQLModel, table=True):
    """
    Sample model representing audio samples uploaded by users.

    Attributes:
        id: Primary key, auto-generated
        user_id: Foreign key reference to User who uploaded the sample
        title: Display name for the sample
        file_url: URL path to the audio file
        instrument: Optional instrument tag for categorization
        mood: Optional mood tag for categorization
        created_at: Timestamp of sample upload
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = Field()
    file_url: str = Field()
    instrument: Optional[str] = Field(default="")
    mood: Optional[str] = Field(default="")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Download(SQLModel, table=True):
    """
    Download model tracking which users downloaded which samples.

    Each record represents a unique user-sample download. The same user
    can only download a sample once (tracked for analytics).

    Attributes:
        id: Primary key, auto-generated
        user_id: Foreign key reference to User who downloaded
        sample_id: Foreign key reference to Sample that was downloaded
        created_at: Timestamp of download
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    sample_id: int = Field(foreign_key="sample.id", index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Upvote(SQLModel, table=True):
    """
    Upvote model tracking which users upvoted which samples.

    Each record represents a user's upvote on a sample. Users can toggle
    their upvote on and off.

    Attributes:
        id: Primary key, auto-generated
        user_id: Foreign key reference to User who upvoted
        sample_id: Foreign key reference to Sample that was upvoted
        created_at: Timestamp of upvote
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    sample_id: int = Field(foreign_key="sample.id", index=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
