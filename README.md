# SmartSamples

A collaborative, free-of-charge platform for music producers to share, discover, and use each other's samples in their own music.

## Features

### MVP Features (Implemented)
- **User Authentication**: Secure registration and login with password hashing
- **Sample Search**: Search for samples with filters for instrument and mood
- **Sample Upload**: Upload MP3, WAV, or OGG audio files with metadata
- **User Profiles**: View user profiles with statistics (uploads, upvotes, downloads)
- **Upvoting System**: Upvote samples you like to support creators
- **Download Tracking**: Track how many times samples are downloaded
- **HTMX Integration**: Dynamic, no-refresh UI interactions

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with SQLModel ORM
- **Frontend**: Jinja2 templates + HTMX for dynamic interactions
- **Styling**: Custom CSS with modern, music-focused aesthetic
- **Authentication**: Session-based auth with itsdangerous + bcrypt password hashing

## Setup Instructions

### Prerequisites
- Python 3.9+
- PostgreSQL database

### Installation

1. **Clone the repository** (or navigate to project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database**:
   ```bash
   # Create a database named 'smartsamples'
   createdb smartsamples

   # Or use psql:
   psql -U postgres
   CREATE DATABASE smartsamples;
   \q
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and update DATABASE_URL and SECRET_KEY
   ```

   Example `.env`:
   ```
   DATABASE_URL=postgresql://your_username:your_password@localhost:5432/smartsamples
   SECRET_KEY=your-generated-secret-key
   ```

6. **Run the application**:
   ```bash
   python main.py
   ```

7. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`


## Future Enhancements

- Advanced audio analysis and recommendation engine
- BPM, key, and metadata extraction from audio files
- Comments system for samples
- User achievements and badges
- Chat/messaging between users
- Sample packs (bundles of samples)
- Multiple audio format support
- Quality assurance for uploads
- Notification system
- Search by audio upload (find similar samples)
