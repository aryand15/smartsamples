"""
SmartSamples - Audio Sample Sharing Platform.

This module implements the main FastAPI application for SmartSamples,
a platform where music producers can upload, discover, and share audio samples.

The application provides:
- User authentication and profile management
- Sample upload and discovery with metadata filtering
- Social features (upvotes, downloads tracking)
- HTMX-powered dynamic UI updates
"""
from fastapi import (
    FastAPI,
    Request,
    Form,
    Depends,
    status,
    UploadFile,
    File,
    HTTPException
)
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import secrets
import os
from pathlib import Path
from sqlmodel import SQLModel, create_engine, Session, select, func
from models import User, Sample, Download, Upvote
from dotenv import load_dotenv
from passlib.context import CryptContext
import shutil
from typing import Optional
import random
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.oggvorbis import OggVorbis

# --- Initialization ---
app = FastAPI()

# Env variables
load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/smartsamples")
engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

def get_session():
    """
    Dependency injection function providing database sessions.

    Yields:
        Session: SQLModel database session with automatic cleanup
    """
    with Session(engine) as session:
        yield session

# Create uploads directory
UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File upload constraints
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB in bytes
MAX_DURATION_SECONDS = 30  # Maximum audio duration in seconds
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.ogg'}

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
serializer = URLSafeTimedSerializer(SECRET_KEY)

def hash_password(password: str) -> str:
    """Hash a plain text password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain text password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_session_token(username: str) -> str:
    """
    Create a signed session token for a username.

    Args:
        username: The username to encode in the token

    Returns:
        str: URL-safe signed token
    """
    return serializer.dumps(username)


def verify_session_token(token: str) -> str | None:
    """
    Verify and decode a session token.

    Args:
        token: The signed session token to verify

    Returns:
        str | None: Username if token is valid, None otherwise
    """
    try:
        # Token expires after 24 hours
        username = serializer.loads(token, max_age=3600 * 24)
        return username
    except (BadSignature, SignatureExpired):
        return None

def get_current_user(
    request: Request,
    session: Session = Depends(get_session)
) -> User | None:
    """
    Get the currently logged-in user from the session cookie.

    Args:
        request: The HTTP request containing cookies
        session: Database session (injected)

    Returns:
        User | None: The authenticated user or None if not logged in
    """
    session_token = request.cookies.get("session")
    if not session_token:
        return None

    username = verify_session_token(session_token)
    if not username:
        return None

    user = session.exec(
        select(User).where(User.username == username)
    ).first()
    return user


def require_auth(
    request: Request,
    session: Session = Depends(get_session)
) -> User:
    """
    Require authentication for protected endpoints.

    Args:
        request: The HTTP request containing cookies
        session: Database session (injected)

    Returns:
        User: The authenticated user

    Raises:
        HTTPException: 401 if user is not authenticated
    """
    user = get_current_user(request, session)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# --- UI Routes (Jinja2 Templates) ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session: Session = Depends(get_session)):
    """Home page - redirects to search if logged in, login if not"""
    user = get_current_user(request, session)
    if user:
        return RedirectResponse(url="/search", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
):
    """Handle login"""
    user = session.exec(select(User).where(User.username == username)).first()

    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid username or password"}
        )

    response = RedirectResponse(url="/search", status_code=status.HTTP_302_FOUND)
    session_token = create_session_token(user.username)
    response.set_cookie(key="session", value=session_token, httponly=True, max_age=86400)

    return response

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Registration page"""
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
):
    """Handle registration"""
    # Check if username or email already exists
    existing_user = session.exec(
        select(User).where((User.username == username) | (User.email == email))
    ).first()

    if existing_user:
        error = "Username already exists" if existing_user.username == username else "Email already exists"
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": error}
        )

    # Create new user
    hashed_password = hash_password(password)
    new_user = User(username=username, email=email, password_hash=hashed_password)
    session.add(new_user)
    session.commit()

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.post("/logout", response_class=HTMLResponse)
async def logout():
    """Handle logout"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("session")
    return response

@app.get("/search", response_class=HTMLResponse)
async def search_page(
    request: Request,
    session: Session = Depends(get_session)
):
    """Sample search page"""
    user = get_current_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse("search.html", {"request": request, "user": user})

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(
    request: Request,
    session: Session = Depends(get_session)
):
    """Sample upload page"""
    user = get_current_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse("upload.html", {"request": request, "user": user})

@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    user_id: int,
    session: Session = Depends(get_session)
):
    """User profile page"""
    current_user = get_current_user(request, session)
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    profile_user = session.get(User, user_id)
    if not profile_user:
        raise HTTPException(status_code=404, detail="User not found")

    return templates.TemplateResponse(
        "profile.html",
        {"request": request, "user": current_user, "profile_user": profile_user}
    )

# --- API Endpoints (JSON) ---

@app.post("/api/auth/logout")
async def api_logout():
    """API endpoint for logout"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("session")
    return response

@app.post("/api/auth/register")
async def api_register(
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session)
):
    """API endpoint for registration"""
    existing_user = session.exec(
        select(User).where((User.username == username) | (User.email == email))
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    hashed_password = hash_password(password)
    new_user = User(username=username, email=email, password_hash=hashed_password)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    return {"user_id": str(new_user.id), "email": new_user.email}

@app.post("/api/sample/{sample_id}/download")
async def api_download_sample(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """Record a sample download"""
    user = require_auth(request, session)

    # Check if already downloaded
    existing_download = session.exec(
        select(Download).where(
            (Download.user_id == user.id) & (Download.sample_id == sample_id)
        )
    ).first()

    if not existing_download:
        download = Download(user_id=user.id, sample_id=sample_id)
        session.add(download)
        session.commit()

    return {"status": "success"}

# --- HTMX Endpoints (HTML Fragments) ---

@app.post("/sample/{sample_id}/upvote", response_class=HTMLResponse)
async def upvote_sample(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """Toggle upvote on a sample"""
    user = get_current_user(request, session)
    if not user:
        return HTMLResponse("<p>Please log in to upvote</p>", status_code=401)

    # Check if already upvoted
    existing_upvote = session.exec(
        select(Upvote).where(
            (Upvote.user_id == user.id) & (Upvote.sample_id == sample_id)
        )
    ).first()

    if existing_upvote:
        # Remove upvote
        session.delete(existing_upvote)
        session.commit()
        upvoted = False
    else:
        # Add upvote
        upvote = Upvote(user_id=user.id, sample_id=sample_id)
        session.add(upvote)
        session.commit()
        upvoted = True

    # Get total upvote count
    upvote_count = session.exec(
        select(func.count(Upvote.id)).where(Upvote.sample_id == sample_id)
    ).one()

    return templates.TemplateResponse(
        "partials/upvote_button.html",
        {
            "request": request,
            "sample_id": sample_id,
            "upvoted": upvoted,
            "upvote_count": upvote_count
        }
    )

@app.post("/sample/{sample_id}/download", response_class=HTMLResponse)
async def download_sample(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """Record a download and return updated download button"""
    user = get_current_user(request, session)
    if not user:
        return HTMLResponse("<p>Please log in to download</p>", status_code=401)

    # Get the sample
    sample = session.get(Sample, sample_id)
    if not sample:
        return HTMLResponse("<p>Sample not found</p>", status_code=404)

    # Check if already downloaded
    existing_download = session.exec(
        select(Download).where(
            (Download.user_id == user.id) & (Download.sample_id == sample_id)
        )
    ).first()

    if not existing_download:
        download = Download(user_id=user.id, sample_id=sample_id)
        session.add(download)
        session.commit()

    # Get total download count
    download_count = session.exec(
        select(func.count(Download.id)).where(Download.sample_id == sample_id)
    ).one()

    return templates.TemplateResponse(
        "partials/download_button.html",
        {
            "request": request,
            "sample_id": sample_id,
            "file_url": sample.file_url,
            "download_count": download_count
        }
    )

@app.post("/recommend-samples", response_class=HTMLResponse)
async def recommend_samples(
    request: Request,
    instrument: str = Form(""),
    mood: str = Form(""),
    limit: int = Form(10),
    session: Session = Depends(get_session)
):
    """Get recommended samples based on filters (simplified for MVP)"""
    user = get_current_user(request, session)
    if not user:
        return HTMLResponse("<p>Please log in</p>", status_code=401)

    # Build query with filters
    query = select(Sample)

    if instrument:
        query = query.where(Sample.instrument.contains(instrument))
    if mood:
        query = query.where(Sample.mood.contains(mood))

    samples = session.exec(query.limit(limit)).all()

    # Get upvote counts and check if current user upvoted
    sample_data = []
    for sample in samples:
        upvote_count = session.exec(
            select(func.count(Upvote.id)).where(Upvote.sample_id == sample.id)
        ).one()

        download_count = session.exec(
            select(func.count(Download.id)).where(Download.sample_id == sample.id)
        ).one()

        user_upvoted = session.exec(
            select(Upvote).where(
                (Upvote.user_id == user.id) & (Upvote.sample_id == sample.id)
            )
        ).first() is not None

        sample_user = session.get(User, sample.user_id)

        sample_data.append({
            "sample": sample,
            "upvote_count": upvote_count,
            "download_count": download_count,
            "user_upvoted": user_upvoted,
            "username": sample_user.username if sample_user else "Unknown"
        })

    return templates.TemplateResponse(
        "partials/sample_results.html",
        {"request": request, "samples": sample_data}
    )

@app.get("/sample/{sample_id}/details", response_class=HTMLResponse)
async def sample_details(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """Get sample details modal"""
    user = get_current_user(request, session)
    if not user:
        return HTMLResponse("<p>Please log in</p>", status_code=401)

    sample = session.get(Sample, sample_id)
    if not sample:
        return HTMLResponse("<p>Sample not found</p>", status_code=404)

    uploader = session.get(User, sample.user_id)
    upvote_count = session.exec(
        select(func.count(Upvote.id)).where(Upvote.sample_id == sample.id)
    ).one()

    download_count = session.exec(
        select(func.count(Download.id)).where(Download.sample_id == sample.id)
    ).one()

    return templates.TemplateResponse(
        "partials/sample_details.html",
        {
            "request": request,
            "sample": sample,
            "uploader": uploader,
            "upvote_count": upvote_count,
            "download_count": download_count
        }
    )

@app.post("/upload-sample", response_class=HTMLResponse)
async def upload_sample(
    request: Request,
    title: str = Form(...),
    instrument: str = Form(""),
    mood: str = Form(""),
    sample: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    """
    Upload a new audio sample with metadata.

    Args:
        request: HTTP request
        title: Display name for the sample
        instrument: Optional instrument tag
        mood: Optional mood tag
        sample: Audio file upload
        session: Database session (injected)

    Returns:
        HTMLResponse: Upload result HTML fragment
    """
    user = get_current_user(request, session)
    if not user:
        return HTMLResponse("<p>Please log in</p>", status_code=401)

    # Validate file extension
    file_ext = Path(sample.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "partials/upload_result.html",
            {
                "request": request,
                "error": "Invalid file type. Only MP3, WAV, and OGG allowed."
            }
        )

    # Read file content to validate size
    content = await sample.read()
    if len(content) > MAX_FILE_SIZE:
        return templates.TemplateResponse(
            "partials/upload_result.html",
            {
                "request": request,
                "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
            }
        )

    # Save file temporarily to check duration
    filename = f"{secrets.token_hex(16)}{file_ext}"
    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        buffer.write(content)

    # Validate audio duration
    try:
        audio = MutagenFile(str(file_path))
        if audio is None or not hasattr(audio.info, 'length'):
            # Clean up the file
            file_path.unlink(missing_ok=True)
            return templates.TemplateResponse(
                "partials/upload_result.html",
                {
                    "request": request,
                    "error": "Unable to read audio file. File may be corrupted."
                }
            )

        duration = audio.info.length
        if duration > MAX_DURATION_SECONDS:
            # Clean up the file
            file_path.unlink(missing_ok=True)
            return templates.TemplateResponse(
                "partials/upload_result.html",
                {
                    "request": request,
                    "error": f"Audio too long. Maximum duration is {MAX_DURATION_SECONDS} seconds (file is {duration:.1f} seconds)."
                }
            )
    except Exception as e:
        # Clean up the file on error
        file_path.unlink(missing_ok=True)
        return templates.TemplateResponse(
            "partials/upload_result.html",
            {
                "request": request,
                "error": f"Error processing audio file: {str(e)}"
            }
        )

    # Create sample record
    new_sample = Sample(
        user_id=user.id,
        title=title,
        file_url=f"/static/uploads/{filename}",
        instrument=instrument,
        mood=mood
    )
    session.add(new_sample)
    session.commit()

    return templates.TemplateResponse(
        "partials/upload_result.html",
        {
            "request": request,
            "success": True,
            "message": "Sample uploaded successfully!",
            "user_id": user.id
        }
    )

@app.put("/edit-profile/{user_id}", response_class=HTMLResponse)
async def edit_profile(
    user_id: int,
    request: Request,
    updated_bio: str = Form(...),
    session: Session = Depends(get_session)
):
    """Edit user profile"""
    user = get_current_user(request, session)
    if not user or user.id != user_id:
        return HTMLResponse("<p>Unauthorized</p>", status_code=403)

    user.bio = updated_bio
    session.add(user)
    session.commit()

    return templates.TemplateResponse(
        "partials/bio.html",
        {"request": request, "bio": updated_bio, "user": user}
    )

@app.get("/user/{user_id}/samples", response_class=HTMLResponse)
async def user_samples(
    user_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """Get all samples for a user"""
    current_user = get_current_user(request, session)
    if not current_user:
        return HTMLResponse("<p>Please log in</p>", status_code=401)

    samples = session.exec(select(Sample).where(Sample.user_id == user_id)).all()

    sample_data = []
    for sample in samples:
        upvote_count = session.exec(
            select(func.count(Upvote.id)).where(Upvote.sample_id == sample.id)
        ).one()

        download_count = session.exec(
            select(func.count(Download.id)).where(Download.sample_id == sample.id)
        ).one()

        sample_data.append({
            "sample": sample,
            "upvote_count": upvote_count,
            "download_count": download_count
        })

    return templates.TemplateResponse(
        "partials/user_samples.html",
        {"request": request, "samples": sample_data}
    )


@app.get("/sample/{sample_id}/edit-title-modal", response_class=HTMLResponse)
async def get_edit_title_modal(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """
    Get the edit title modal for a sample.

    Only the sample owner can edit their sample titles.

    Args:
        sample_id: ID of the sample to edit
        request: HTTP request
        session: Database session (injected)

    Returns:
        HTMLResponse: Edit title modal HTML

    Raises:
        HTTPException: 404 if sample not found, 403 if not authorized, 401 if not authenticated
    """
    user = get_current_user(request, session)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if sample.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this sample")

    return templates.TemplateResponse(
        "partials/edit_title_modal.html",
        {"request": request, "sample": sample}
    )


@app.put("/sample/{sample_id}/title", response_class=HTMLResponse)
async def update_sample_title(
    sample_id: int,
    request: Request,
    updated_title: str = Form(...),
    session: Session = Depends(get_session)
):
    """
    Update a sample's title.

    Only the sample owner can update their sample titles.

    Args:
        sample_id: ID of the sample to update
        request: HTTP request
        updated_title: New title for the sample
        session: Database session (injected)

    Returns:
        HTMLResponse: Updated sample card HTML

    Raises:
        HTTPException: 404 if sample not found, 403 if not authorized, 401 if not authenticated
    """
    user = get_current_user(request, session)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if sample.user_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this sample")

    # Validate title
    updated_title = updated_title.strip()
    if not updated_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")

    # Update the title
    sample.title = updated_title
    session.add(sample)
    session.commit()
    session.refresh(sample)

    # Get upvote and download counts for the updated card
    upvote_count = session.exec(
        select(func.count(Upvote.id)).where(Upvote.sample_id == sample.id)
    ).one()

    download_count = session.exec(
        select(func.count(Download.id)).where(Download.sample_id == sample.id)
    ).one()

    return templates.TemplateResponse(
        "partials/single_sample_card.html",
        {
            "request": request,
            "sample": sample,
            "upvote_count": upvote_count,
            "download_count": download_count
        }
    )


@app.delete("/sample/{sample_id}", response_class=HTMLResponse)
async def delete_sample(
    sample_id: int,
    request: Request,
    session: Session = Depends(get_session)
):
    """
    Delete a sample and its associated file.

    Only the sample owner can delete their samples.

    Args:
        sample_id: ID of the sample to delete
        request: HTTP request
        session: Database session (injected)

    Returns:
        HTMLResponse: Empty response on success

    Raises:
        HTTPException: 404 if sample not found, 403 if not authorized
    """
    user = get_current_user(request, session)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get the sample
    sample = session.get(Sample, sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    # Check ownership
    if sample.user_id != user.id:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to delete this sample"
        )

    # Delete associated upvotes
    upvotes = session.exec(
        select(Upvote).where(Upvote.sample_id == sample_id)
    ).all()
    for upvote in upvotes:
        session.delete(upvote)

    # Delete associated downloads
    downloads = session.exec(
        select(Download).where(Download.sample_id == sample_id)
    ).all()
    for download in downloads:
        session.delete(download)

    # Delete the file from disk if it exists
    file_path = Path(sample.file_url.lstrip('/'))
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError as e:
            # Log error but continue with database deletion
            print(f"Error deleting file {file_path}: {e}")

    # Delete the sample record
    session.delete(sample)
    session.commit()

    # Return empty response for HTMX to remove the element
    return HTMLResponse("", status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
