"""
Database setup script for SmartSamples
Run this to create the database tables and optionally seed with test data
"""

import os
from sqlmodel import SQLModel, create_engine, Session
from models import User, Sample, Download, Upvote
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_tables():
    """Create all database tables"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL, echo=True)

    print("Creating tables...")
    SQLModel.metadata.create_all(engine)
    print("Tables created successfully!")

    return engine

def seed_test_data(engine):
    """Add test data to the database"""
    with Session(engine) as session:
        print("\nSeeding test data...")

        # Create test users
        user1 = User(
            username="producer1",
            email="producer1@example.com",
            password_hash=pwd_context.hash("password123"),
            bio="I love creating chill beats and ambient soundscapes."
        )

        user2 = User(
            username="beatmaker",
            email="beatmaker@example.com",
            password_hash=pwd_context.hash("password123"),
            bio="Hip-hop producer specializing in hard-hitting drums and 808s."
        )

        session.add(user1)
        session.add(user2)
        session.commit()
        session.refresh(user1)
        session.refresh(user2)

        print(f"Created users: {user1.username}, {user2.username}")

        # Note: You'll need to manually add sample files to static/uploads/
        # or create placeholder entries

        print("\nTest data seeded successfully!")
        print("\nTest credentials:")
        print("  Username: producer1, Password: password123")
        print("  Username: beatmaker, Password: password123")

if __name__ == "__main__":
    print("=== SmartSamples Database Setup ===\n")

    # Create tables
    engine = create_tables()

    # Ask if user wants to seed test data
    seed = input("\nWould you like to seed test data? (y/n): ").strip().lower()

    if seed == 'y':
        seed_test_data(engine)

    print("\n=== Setup Complete ===")
    print("You can now run the application with: python main.py")
