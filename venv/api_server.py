"""
Enhanced FastAPI server with proper audio file handling and background music mixing
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import hashlib
import re
import jwt
import os
import tempfile
import logging
import base64
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
import requests
from pydub import AudioSegment
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure logging exactly like Streamlit
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure FFmpeg paths exactly like Streamlit
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'C:\\ffmpeg\\bin\\ffmpeg.exe')
FFPROBE_PATH = os.getenv('FFPROBE_PATH', 'C:\\ffmpeg\\bin\\ffprobe.exe')
FREESOUND_API_KEY = os.getenv('FREESOUND_API_KEY')

# Configure pydub to use FFmpeg exactly like Streamlit
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

app = FastAPI(title="AI Tools Hub API", version="1.0.0")

# Create directories for file storage
os.makedirs("temp_audio", exist_ok=True)
os.makedirs("temp_images", exist_ok=True)

# Mount static files
app.mount("/audio", StaticFiles(directory="temp_audio"), name="audio")
app.mount("/images", StaticFiles(directory="temp_images"), name="images")

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys and Configuration - using exact same variable names as Streamlit
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "unified_ai")
GOOGLE_API_KEY = os.getenv("google_api")  # Note: using "google_api" like Streamlit
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
IMAGEKIT_PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
IMAGEKIT_PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
IMAGEKIT_URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")

# Configure services
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize ImageKit
imagekit = None
if all([IMAGEKIT_PRIVATE_KEY, IMAGEKIT_PUBLIC_KEY, IMAGEKIT_URL_ENDPOINT]):
    imagekit = ImageKit(
        private_key=IMAGEKIT_PRIVATE_KEY,
        public_key=IMAGEKIT_PUBLIC_KEY,
        url_endpoint=IMAGEKIT_URL_ENDPOINT
    )

# Database connection
db = None
try:
    client = MongoClient(MONGO_URI)
    db_instance = client[DATABASE_NAME]
    # Test connection
    db_instance.command('ping')
    db = {
        "users": db_instance["users"],
        "history": db_instance["history"]
    }
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str

class ActivityRequest(BaseModel):
    activity_type: str

class TTSRequest(BaseModel):
    text: str
    voice: str
    music_search: Optional[str] = ""

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

class UserResponse(BaseModel):
    username: str
    email: str

# Utility functions
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def get_current_user(username: str = Depends(verify_token)):
    """Get current user from database"""
    if not db:
        raise HTTPException(status_code=500, detail="Database connection not available")
    
    user = db["users"].find_one({"username": username})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def store_activity(username: str, activity_type: str, details: dict, output_url: Optional[str] = None):
    """Store user activity in database"""
    try:
        if not db:
            logger.error("Database connection not available")
            return False
            
        activity = {
            "username": username,
            "activity_type": activity_type,
            "details": details,
            "output_url": output_url,
            "timestamp": datetime.utcnow()
        }
        
        # Print the activity being stored for debugging (like Streamlit)
        logger.info(f"Storing activity: {activity}")
        
        result = db["history"].insert_one(activity)
        logger.info(f"Activity stored with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing activity: {e}")
        return False

def get_gemini_response(question: str) -> str:
    """Get response from Google Gemini AI"""
    try:
        if not GOOGLE_API_KEY:
            raise Exception("Google API key not configured")
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# Text-to-Speech Functions - copied exactly from Streamlit

def get_available_voices(api_key, region):
    """Get available voices - copied exactly from Streamlit"""
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    voices_result = synthesizer.get_voices_async().get()
    
    if voices_result.reason == speechsdk.ResultReason.VoicesListRetrieved:
        english_voices = [
            (voice.short_name, voice.short_name.split('-')[-1])
            for voice in voices_result.voices if voice.locale.startswith('en')
        ]
        return english_voices
    return []

def text_to_speech_azure(api_key, region, text, voice, filename="output.wav"):
    """Convert text to speech - copied exactly from Streamlit"""
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    
    # Create full path in temp_audio directory
    filepath = os.path.join("temp_audio", filename)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filepath)
    
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return filepath  # Return full path
    return None

def setup_ffmpeg():
    """Setup FFmpeg - copied exactly from Streamlit"""
    try:
        AudioSegment.silent(duration=100)
        logger.debug("FFmpeg test successful")
        return True
    except Exception as e:
        logger.error(f"FFmpeg test failed: {str(e)}")
        return False

def search_background_music(query, freesound_api_key):
    """Search background music - copied exactly from Streamlit"""
    try:
        headers = {
            'Authorization': f'Token {freesound_api_key}'
        }
        
        params = {
            'query': query,
            'filter': 'duration:[1 TO 60]',
            'fields': 'id,name,duration,previews',
            'page_size': 5
        }
        
        response = requests.get(
            'https://freesound.org/apiv2/search/text/',
            headers=headers,
            params=params
        )
        
        response.raise_for_status()
        results = response.json().get('results', [])
        return [result for result in results if 'previews' in result and 'preview-hq-mp3' in result['previews']]
        
    except Exception as e:
        logger.error(f"Error in search_background_music: {str(e)}")
        return []

def download_and_process_audio(preview_url, volume_reduction=15):
    """Download and process audio - with increased volume reduction"""
    temp_path = None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"Downloading audio from: {preview_url}")
        response = requests.get(preview_url, headers=headers, timeout=15)
        if response.status_code != 200:
            logger.error(f"Failed to download audio: HTTP {response.status_code}")
            return None
            
        logger.info(f"Downloaded {len(response.content)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
            
        try:
            logger.info("Loading MP3 with pydub...")
            audio = AudioSegment.from_mp3(temp_path)
            logger.info(f"Original audio: {len(audio)}ms, {audio.dBFS}dB")
            
            # Reduce volume significantly
            processed_audio = audio - volume_reduction
            logger.info(f"Processed audio: {len(processed_audio)}ms, {processed_audio.dBFS}dB")
            
            return processed_audio
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error in download_and_process_audio: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def mix_audio(speech_path, background_music, output_path="final_output.wav"):
    """Mix audio - corrected version with proper volume balancing"""
    try:
        logger.info(f"Loading speech from: {speech_path}")
        speech = AudioSegment.from_wav(speech_path)
        logger.info(f"Speech duration: {len(speech)}ms, volume: {speech.dBFS}dB")
        
        logger.info(f"Background music duration: {len(background_music)}ms, volume: {background_music.dBFS}dB")
        
        # Loop background music to match speech duration
        original_bg_length = len(background_music)
        while len(background_music) < len(speech):
            background_music = background_music + background_music
        
        # Trim background music to speech length
        background_music = background_music[:len(speech)]
        logger.info(f"Background music adjusted to: {len(background_music)}ms")
        
        # Reduce background music volume significantly more to ensure speech is prominent
        background_music = background_music - 10  
        logger.info(f"Background music volume after reduction: {background_music.dBFS}dB")
        
        # Boost speech volume slightly if needed
        if speech.dBFS < -10:  # If speech is too quiet
            speech = speech + 3  # Boost by 3dB
            logger.info(f"Speech volume boosted to: {speech.dBFS}dB")
        
        # Mix: background music as base, speech overlaid on top
        final_audio = background_music.overlay(speech)
        logger.info(f"Final mixed audio duration: {len(final_audio)}ms, volume: {final_audio.dBFS}dB")
        
        # Create full path in temp_audio directory
        final_path = os.path.join("temp_audio", output_path)
        final_audio.export(final_path, format="wav")
        logger.info(f"Mixed audio exported to: {final_path}")
        
        return final_path
    except Exception as e:
        logger.error(f"Error mixing audio: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Tools Hub API is running", "status": "healthy"}

@app.post("/api/signup", response_model=Token)
async def signup(user: UserCreate):
    """User registration endpoint"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Validate password
        valid, message = validate_password(user.password)
        if not valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Check if user exists
        existing_user = db["users"].find_one({
            "$or": [{"email": user.email}, {"username": user.username}]
        })
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Email or username already exists")
        
        # Create user
        user_data = {
            "email": user.email,
            "username": user.username,
            "password": hash_password(user.password),
            "created_at": datetime.utcnow()
        }
        
        result = db["users"].insert_one(user_data)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"username": user.username, "email": user.email}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    """User login endpoint"""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Find user
        db_user = db["users"].find_one({
            "username": user.username,
            "password": hash_password(user.password)
        })
        
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {"username": db_user["username"], "email": db_user["email"]}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(username=current_user["username"], email=current_user["email"])

@app.post("/api/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Handle chat messages"""
    try:
        response = get_gemini_response(request.message)
        
        # Store activity
        store_activity(
            current_user["username"],
            "chat",
            {"user_message": request.message, "bot_response": response}
        )
        
        return {"response": response}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")

@app.post("/api/activities")
async def get_activities(request: ActivityRequest, current_user: dict = Depends(get_current_user)):
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Database connection not available")
        
        # Map display names to database activity types - matching Streamlit
        activity_type_map = {
            'Chatbot': 'chat',
            'Photo Editor': 'photo_edit',
            'Text to Speech': 'text_to_speech',
            'File Upload': 'file_upload'
        }
        
        db_activity_type = activity_type_map.get(request.activity_type, request.activity_type.lower())
        
        query = {
            "username": current_user["username"],
            "activity_type": db_activity_type
        }
        
        activities = list(db["history"].find(query, {"_id": 0}).sort("timestamp", -1).limit(20))
        
        for activity in activities:
            if isinstance(activity.get("timestamp"), datetime):
                activity["timestamp"] = activity["timestamp"].isoformat()
            if "id" not in activity:
                activity["id"] = str(activity.get("_id", f"{activity['timestamp']}_{activity['activity_type']}"))
        
        return {"activities": activities}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Activities error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving activities")

@app.post("/api/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    transformation: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and transform image using ImageKit"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (10MB limit)
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        logger.info(f"Processing file: {file.filename}, size: {len(file_content)} bytes, type: {file.content_type}")
        
        if not imagekit:
            # Demo mode - save locally
            filename = f"demo_{int(datetime.now().timestamp())}_{file.filename}"
            filepath = os.path.join("temp_images", filename)
            
            with open(filepath, "wb") as f:
                f.write(file_content)
            
            demo_url = f"/images/{filename}"
            
            store_activity(
                current_user["username"],
                "photo_edit",
                {"transformation": transformation, "original_filename": file.filename},
                demo_url
            )
            
            return {
                "original_url": demo_url,
                "transformed_url": demo_url,
                "file_path": f"demo/{file.filename}",
                "message": "Demo mode: ImageKit not configured"
            }
        
        # Create temporary file (similar to Streamlit approach)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(file_content)
            temp_path = tmp_file.name
        
        try:
            # Upload to ImageKit using file path (like Streamlit)
            with open(temp_path, "rb") as temp_file:
                upload_result = imagekit.upload_file(
                    file=temp_file,
                    file_name=f"{current_user['username']}_{int(datetime.now().timestamp())}_{file.filename}",
                    options=UploadFileRequestOptions(tags=[transformation.lower()])
                )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if not upload_result.url:
                raise HTTPException(status_code=500, detail="Failed to upload image to ImageKit")
            
            logger.info(f"Image uploaded successfully: {upload_result.url}")
            
            # Apply transformation (matching Streamlit logic)
            transform_map = {
                "blur": {"blur": "20"},
                "grayscale": {"effect_gray": "-"},
                "sepia": {"effect_sepia": "50"},
                "flip-h": {"flip": "h"},
                "flip-v": {"flip": "v"}
            }
            
            transform = transform_map.get(transformation.lower(), {})
            
            if transform and upload_result.file_path:
                # Create transformed URL using file_path (like Streamlit)
                transformed_url = imagekit.url({
                    "path": upload_result.file_path,
                    "transformation": [transform]
                })
                logger.info(f"Transformation applied: {transformed_url}")
            else:
                transformed_url = upload_result.url
                logger.info(f"No transformation applied, using original: {transformed_url}")
            
            # Store activity
            store_activity(
                current_user["username"],
                "photo_edit",
                {"transformation": transformation, "original_filename": file.filename},
                transformed_url
            )
            
            return {
                "original_url": upload_result.url,
                "transformed_url": transformed_url,
                "file_path": upload_result.file_path,
                "message": "Image processed successfully"
            }
            
        except Exception as imagekit_error:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"ImageKit processing error: {imagekit_error}")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {str(imagekit_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")

@app.get("/api/voices")
async def get_voices():
    """Get available speech voices"""
    try:
        if not all([AZURE_SPEECH_KEY, AZURE_SPEECH_REGION]):
            return {"voices": [
                {"name": "en-US-JennyNeural", "display_name": "Jenny"},
                {"name": "en-US-GuyNeural", "display_name": "Guy"},
                {"name": "en-US-AriaNeural", "display_name": "Aria"},
                {"name": "en-US-DavisNeural", "display_name": "Davis"},
            ]}
        
        voices = get_available_voices(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
        voice_list = []
        for full_name, display_name in voices:
            voice_list.append({
                "name": full_name,
                "display_name": display_name.replace("Neural", "").strip()
            })
        
        return {"voices": voice_list[:10]}  # Limit to 10 voices
    except Exception as e:
        logger.error(f"Get voices error: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving voices")

@app.post("/api/text-to-speech")
async def text_to_speech(request: TTSRequest, current_user: dict = Depends(get_current_user)):
    """Convert text to speech with background music - following Streamlit workflow exactly"""
    try:
        # Check required API keys exactly like Streamlit
        if not all([AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, FREESOUND_API_KEY]):
            raise HTTPException(status_code=500, detail="Missing required API keys. Please check your environment variables.")
        
        # Check FFmpeg exactly like Streamlit
        if not setup_ffmpeg():
            raise HTTPException(status_code=500, detail="FFmpeg configuration failed. Please ensure FFmpeg is properly installed.")
        
        # Validate input exactly like Streamlit
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Please enter text to convert to speech.")
            
        if not request.music_search.strip():
            raise HTTPException(status_code=400, detail="Please enter keywords for background music.")
        
        logger.info("Step 1: Generating speech...")
        # Step 1: Generate speech using default filename like Streamlit
        speech_file = text_to_speech_azure(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, request.text, request.voice)
        if not speech_file:
            raise HTTPException(status_code=500, detail="Failed to generate speech audio.")
        
        logger.info("Speech generated successfully!")
        
        logger.info("Step 2: Searching for background music...")
        # Step 2: Search for background music
        music_results = search_background_music(request.music_search, FREESOUND_API_KEY)
        if not music_results:
            raise HTTPException(status_code=404, detail="No music found matching your criteria.")
            
        logger.info("Found matching background music!")
        
        # In Streamlit, user selects track. For API, auto-select first one
        selected_music = music_results[0]
        preview_url = selected_music['previews']['preview-hq-mp3']
        selected_track_name = f"{selected_music['name']} ({selected_music['duration']:.1f}s)"
        
        logger.info("Step 3: Processing and mixing audio...")
        # Step 3: Process and mix audio
        background_music = download_and_process_audio(preview_url)
        if not background_music:
            raise HTTPException(status_code=500, detail="Failed to process background music.")
            
        # Create unique filename for final output like Streamlit
        timestamp = int(datetime.now().timestamp())
        final_output_filename = f"final_output_{timestamp}.wav"
        
        # Mix audio using unique filename
        final_output = mix_audio(speech_file, background_music, final_output_filename)
        if not final_output:
            raise HTTPException(status_code=500, detail="Failed to mix audio.")

        logger.info("Audio processing complete!")
        
        # Get just the filename for the URL
        final_filename = os.path.basename(final_output)
        audio_url = f"/audio/{final_filename}"
        
        # Store activity exactly like Streamlit
        store_activity(
            current_user["username"],
            "text_to_speech",
            {
                "text": request.text,
                "voice": request.voice,
                "background_music": selected_track_name
            }
        )
        
        # Cleanup temporary files exactly like Streamlit
        try:
            if os.path.exists(speech_file):
                os.remove(speech_file)
        except Exception as e:
            logger.warning(f"Failed to cleanup speech file: {str(e)}")
        
        return {
            "audio_url": audio_url,
            "background_music": {
                "name": selected_music["name"],
                "duration": selected_music["duration"]
            },
            "filename": final_filename,
            "has_background_music": True,
            "selected_track": selected_track_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech service failed: {str(e)}")

@app.get("/api/download-audio/{filename}")
async def download_audio(filename: str, current_user: dict = Depends(get_current_user)):
    """Download audio file"""
    try:
        filepath = os.path.join("temp_audio", filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            filepath,
            media_type="audio/wav",
            filename=f"speech_with_music_{filename}"
        )
    except Exception as e:
        logger.error(f"Download audio error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download audio")

@app.post("/api/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload general file"""
    try:
        # Validate file size (10MB limit)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Store file metadata
        file_details = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(file_content)
        }
        
        store_activity(
            current_user["username"],
            "file_upload",
            file_details
        )
        
        return {"message": "File uploaded successfully", "details": file_details}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.post("/api/debug/test-audio-mixing")
async def test_audio_mixing(current_user: dict = Depends(get_current_user)):
    """Debug endpoint to test audio mixing with sample files"""
    try:
        # Create a test speech file
        test_speech_path = os.path.join("temp_audio", "test_speech.wav")
        
        # Generate a simple test speech
        if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
            speech_file = text_to_speech_azure(
                AZURE_SPEECH_KEY, 
                AZURE_SPEECH_REGION, 
                "This is a test of audio mixing with background music.", 
                "en-US-JennyNeural",
                "test_speech.wav"
            )
        else:
            return {"error": "Azure Speech credentials not available"}
        
        if not speech_file:
            return {"error": "Failed to generate test speech"}
        
        # Search for test background music
        music_results = search_background_music("piano calm", FREESOUND_API_KEY)
        if not music_results:
            return {"error": "No background music found"}
        
        # Download and process first result
        preview_url = music_results[0]['previews']['preview-hq-mp3']
        background_music = download_and_process_audio(preview_url)#, volume_reduction=10
        
        if not background_music:
            return {"error": "Failed to process background music"}
        
        # Test mixing
        mixed_file = mix_audio(speech_file, background_music, "test_mixed.wav")
        
        if mixed_file:
            return {
                "success": True,
                "mixed_file": f"/audio/test_mixed.wav",
                "speech_file": f"/audio/test_speech.wav",
                "message": "Audio mixing test completed successfully"
            }
        else:
            return {"error": "Audio mixing failed"}
            
    except Exception as e:
        logger.error(f"Test audio mixing error: {e}")
        return {"error": str(e)}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="debug"
    )
