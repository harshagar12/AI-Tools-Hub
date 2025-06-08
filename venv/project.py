import streamlit as st
import os
import textwrap
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from PIL import Image
import tempfile
from io import BytesIO
import requests
from pydub import AudioSegment
import logging
import hashlib
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging for text-to-speech
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Additional environment variables for FFmpeg
FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'C:\\ffmpeg\\bin\\ffmpeg.exe')
FFPROBE_PATH = os.getenv('FFPROBE_PATH', 'C:\\ffmpeg\\bin\\ffprobe.exe')
FREESOUND_API_KEY = os.getenv('FREESOUND_API_KEY')

# Configure pydub to use FFmpeg
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "unified_ai")
db = None

# Configure API Keys
genai.configure(api_key=os.getenv("google_api"))

# ImageKit setup
PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")
PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")
URL_ENDPOINT = os.getenv("IMAGEKIT_URL_ENDPOINT")

# Azure Speech Service setup
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# User Authentication Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def validate_password(password):
    # At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def init_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        # Test the connection
        db.command('ping')
        print("Successfully connected to MongoDB")
        return {
            "users": db["users"],
            "history": db["history"]
        }
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def store_activity(db, username, activity_type, details, output_url=None):
    try:
        if not db:
            print("Database connection not available")
            return False
            
        activity = {
            "username": username,
            "activity_type": activity_type,
            "details": details,
            "output_url": output_url,
            "timestamp": datetime.utcnow()
        }
        
        # Print the activity being stored for debugging
        print(f"Storing activity: {activity}")
        
        result = db["history"].insert_one(activity)
        print(f"Activity stored with ID: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"Error storing activity: {e}")
        return False

def display_history(db, activity_type):
    try:
        if not db:
            st.error("Database connection not available")
            return
            
        st.subheader("Recent Activity")
        
        # Print debug information
        print(f"Fetching history for user: {st.session_state.get('username')}")
        print(f"Activity type: {activity_type}")
        
        # Ensure we're using the correct activity type format
        normalized_activity_type = activity_type.lower().replace(" ", "_")
        
        query = {
            "username": st.session_state.get('username'),
            "activity_type": normalized_activity_type
        }
        
        print(f"Query: {query}")
        
        history = list(db["history"].find(query).sort("timestamp", -1).limit(5))
        print(f"Found {len(history)} history items")
        
        if not history:
            st.info("No recent activity found")
            return
        
        for item in history:
            try:
                timestamp = item.get('timestamp', datetime.utcnow())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                    
                with st.expander(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                    if normalized_activity_type == "chat":
                        if 'details' in item and 'user_message' in item['details']:
                            st.markdown("**Your message:**")
                            st.markdown(f"_{item['details']['user_message']}_")
                            st.markdown("**Bot response:**")
                            st.markdown(item['details']['bot_response'])
                        else:
                            st.warning("Incomplete chat record")
                            
                    elif normalized_activity_type == "photo_editor":
                        if 'details' in item:
                            st.markdown(f"**Transformation:** {item['details'].get('transformation', 'N/A')}")
                            if item.get('output_url'):
                                st.image(item['output_url'])
                                
                    elif normalized_activity_type == "text_to_speech":
                        if 'details' in item:
                            st.markdown(f"**Text:** {item['details'].get('text', 'N/A')}")
                            st.markdown(f"**Voice:** {item['details'].get('voice', 'N/A')}")
                            
                    else:
                        st.json(item.get('details', {}))
                        
            except Exception as e:
                print(f"Error displaying history item: {e}")
                continue
                
    except Exception as e:
        print(f"Error in display_history: {e}")
        st.error("Error loading history")

def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(question)
    return response.text

def get_available_voices(api_key, region):
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
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)
    
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return filename
    return None

def create_header():
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #1e3799;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #0984e3;
        }
        .main-content {
            margin-top: 2rem;
            padding: 1rem;
        }
        .history-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            border: 1px solid #dee2e6;
        }
        .history-title {
            color: #1e3799;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: white;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #1e3799;
        }
        .bot-message {
            background-color: #f5f5f5;
            border-left: 4px solid #0984e3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.get('logged_in'):
        st.title("AI Tools Hub")
        
        # Navigation menu
        menu_options = ["Chatbot", "Photo Editor", "Text to Speech", "File Upload"]
        cols = st.columns(len(menu_options) + 1)  # +1 for logout
        
        for idx, option in enumerate(menu_options):
            if cols[idx].button(option):
                st.session_state['current_page'] = option
                st.rerun()
        
        if cols[-1].button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state['page'] = 'login'
            st.rerun()

def display_history(db, activity_type):
    try:
        if not db:
            st.error("Database connection not available")
            return
            
        st.subheader("Recent Activity")
        
        # Map display names to database activity types
        activity_type_map = {
            'Chatbot': 'chat',
            'Photo Editor': 'photo_edit',
            'Text to Speech': 'text_to_speech',
            'File Upload': 'file_upload'
        }
        
        # Get the correct activity type from the map
        db_activity_type = activity_type_map.get(activity_type, activity_type.lower())
        
        # Query the database
        query = {
            "username": st.session_state.get('username'),
            "activity_type": db_activity_type
        }
        
        history = list(db["history"].find(query).sort("timestamp", -1).limit(5))
        
        if not history:
            st.info(f"No recent activity found for {activity_type}")
            return
        
        # Display history items
        for item in history:
            timestamp = item.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
                
            with st.expander(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
                if db_activity_type == "chat":
                    if 'details' in item and 'user_message' in item['details']:
                        st.markdown("""
                            <div style='background-color: #000000; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                                <strong>Your message:</strong><br/>
                                {msg}
                            </div>
                            <div style='background-color: #000000; padding: 10px; border-radius: 5px;'>
                                <strong>Bot response:</strong><br/>
                                {resp}
                            </div>
                        """.format(
                            msg=item['details']['user_message'],
                            resp=item['details']['bot_response']
                        ), unsafe_allow_html=True)
                        
                elif db_activity_type == "photo_edit":
                    st.markdown(f"**Transformation:** {item['details'].get('transformation', 'N/A')}")
                    if 'output_url' in item and item['output_url']:
                        st.image(item['output_url'])
                        
                elif db_activity_type == "text_to_speech":
                    st.markdown(f"**Text:** {item['details'].get('text', 'N/A')}")
                    st.markdown(f"**Voice:** {item['details'].get('voice', 'N/A')}")
                    if 'background_music' in item['details']:
                        st.markdown(f"**Background Music:** {item['details']['background_music']}")
                        
                elif db_activity_type == "file_upload":
                    st.markdown("**File Details:**")
                    st.json(item['details'])
                    
    except Exception as e:
        st.error(f"Error displaying history: {str(e)}")


def signup_page(db):
    st.title("AI Tools Hub - Sign Up")
    
    with st.form("signup_form"):
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if not email or not username or not password:
                st.error("All fields are required!")
                return
                
            if not validate_email(email):
                st.error("Invalid email format!")
                return
                
            if password != confirm_password:
                st.error("Passwords don't match!")
                return
                
            valid_password, message = validate_password(password)
            if not valid_password:
                st.error(message)
                return
                
            # Check if user exists
            existing_user = db["users"].find_one({"$or": [
                {"email": email},
                {"username": username}
            ]})
            
            if existing_user:
                st.error("Email or username already exists!")
                return
                
            # Create new user
            user_data = {
                "email": email,
                "username": username,
                "password": hash_password(password),
                "created_at": datetime.utcnow()
            }
            
            db["users"].insert_one(user_data)
            st.success("Account created successfully! Please log in.")

def login_page(db):
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.title("Welcome Back!")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password!")
                return
                
            user = db["users"].find_one({
                "username": username,
                "password": hash_password(password)
            })
            
            if user:
                st.session_state['username'] = username
                st.session_state['logged_in'] = True
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
            pass

    if st.button("Don't have an account? Sign Up"):
        st.session_state['page'] = 'signup'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# def chatbot_page(db):
#     st.header("AI Chatbot")
#     user_input = st.text_input("Your message:")
    
#     if st.button("Send"):
#         if user_input:
#             with st.spinner("Getting response..."):
#                 try:
#                     response = get_gemini_response(user_input)
                    
#                     # Display the current chat
#                     st.markdown("""
#                         <div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
#                             <strong>Your message:</strong><br/>
#                             {msg}
#                         </div>
#                         <div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px;'>
#                             <strong>Bot response:</strong><br/>
#                             {resp}
#                         </div>
#                     """.format(
#                         msg=user_input,
#                         resp=response
#                     ), unsafe_allow_html=True)
                    
#                     # Store in history
#                     success = store_activity(
#                         db,
#                         st.session_state.get('username'),
#                         "chat",  # This should match the db_activity_type in display_history
#                         {"user_message": user_input, "bot_response": response}
#                     )
                    
#                     if success:
#                         st.rerun()  # Refresh to show updated history
                        
#                 except Exception as e:
#                     st.error(f"An error occurred: {str(e)}")
def chatbot_page(db):
    st.header("AI Chatbot")
    user_input = st.text_input("Your message:")
    
    if st.button("Send"):
        if user_input:
            with st.spinner("Getting response..."):
                try:
                    response = get_gemini_response(user_input)
                    
                    # Display the current chat
                    st.markdown("""
                        <div style='background-color:rgb(0, 0, 0); padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <strong>Your message:</strong><br/>
                            {msg}
                        </div>
                        <div style='background-color:rgb(0, 0, 0); padding: 10px; border-radius: 5px;'>
                            <strong>Bot response:</strong><br/>
                            {resp}
                        </div>
                    """.format(
                        msg=user_input,
                        resp=response
                    ), unsafe_allow_html=True)
                    
                    # Store in history after displaying
                    success = store_activity(
                        db,
                        st.session_state.get('username'),
                        "chat",  # This should match the db_activity_type in display_history
                        {"user_message": user_input, "bot_response": response}
                    )
                    
                    if not success:
                        st.error("Failed to store chat history")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            

def photo_editor_page(db):
    st.title("Photo Editor")
    
    if not all([PRIVATE_KEY, PUBLIC_KEY, URL_ENDPOINT]):
        st.error("ImageKit environment variables not set!")
        return
        
    imagekit = ImageKit(
        private_key=PRIVATE_KEY,
        public_key=PUBLIC_KEY,
        url_endpoint=URL_ENDPOINT
    )
    
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, "Original Image")
        
        option = st.selectbox(
            "Select transformation",
            ["None", "Blur", "Grayscale", "Text", "Flip H", "Flip V", "Sepia"]
        )

        transform = {}
        if option == "Blur":
            transform = {"blur": "20"}
        elif option == "Grayscale":
            transform = {"effect_gray": "-"}
        elif option == "Text":
            text = st.text_input("Text:")
            if text:
                transform = {
                    "overlay_text": text,
                    "overlay_text_font_size": "30",
                    "overlay_text_color": "FFFFFF"
                }
        elif option == "Flip H":
            transform = {"flip": "h"}
        elif option == "Flip V":
            transform = {"flip": "v"}
        elif option == "Sepia":
            transform = {"effect_sepia": "50"}

        if st.button("Upload and Transform"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                with open(temp_path, "rb") as file:
                    upload = imagekit.upload_file(
                        file=file,
                        file_name=f"{st.session_state['username']}_{uploaded_file.name}",
                        options=UploadFileRequestOptions(tags=[option.lower()])
                    )

                os.unlink(temp_path)

                if upload.url and transform:
                    transformed_url = imagekit.url({
                        "path": upload.file_path,
                        "transformation": [transform]
                    })
                    
                    st.image(transformed_url, "Transformed Image")
                    st.success("Transform successful!")
                    
                    st.write("Transformed Image URL:")
                    st.code(transformed_url)
                    store_activity(
                        db,
                        st.session_state['username'],
                        "photo_edit",
                        {"transformation": option},
                        transformed_url
                    )
                    
                    response = requests.get(transformed_url)
                    if response.status_code == 200:
                        if st.download_button(
                            "Download Transformed Image",
                            BytesIO(response.content).getvalue(),
                            f"transformed_{uploaded_file.name}",
                            mime=f"image/{image.format.lower()}"
                        ):
                            store_activity(
                                db,
                                st.session_state['username'],
                                "photo_download",
                                {"transformation": option},
                                transformed_url
                            )
                            st.success("Downloaded successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

def text_to_speech_page(db):
    st.title("Text to Speech with Background Music")
    
    if not all([AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, FREESOUND_API_KEY]):
        st.error("Missing required API keys. Please check your environment variables.")
        return
    
    if not setup_ffmpeg():
        st.error("""
        FFmpeg configuration failed. Please ensure:
        1. FFmpeg is properly installed
        2. Correct paths are set in your .env file
        """)
        return
    
    voices = get_available_voices(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
    if voices:
        voice_display_names = {display_name.replace("Neural", "").strip(): full_name 
                             for full_name, display_name in voices}
        
        selected_display_name = st.selectbox("Select Voice:", list(voice_display_names.keys()))
        selected_voice = voice_display_names[selected_display_name]
        
        text = st.text_area("Enter text to convert to speech:")
        music_prompt = st.text_input("Enter keywords for background music (e.g., 'calm piano background'):")
        
        if st.button("Convert to Speech with Music"):
            if not text.strip():
                st.error("Please enter text to convert to speech.")
                return
                
            if not music_prompt.strip():
                st.error("Please enter keywords for background music.")
                return

            try:
                with st.spinner("Step 1: Generating speech..."):
                    speech_file = text_to_speech_azure(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, text, selected_voice)
                    if not speech_file:
                        st.error("Failed to generate speech audio.")
                        return
                    st.success("Speech generated successfully!")

                with st.spinner("Step 2: Searching for background music..."):
                    music_results = search_background_music(music_prompt, FREESOUND_API_KEY)
                    if not music_results:
                        st.error("No music found matching your criteria.")
                        return
                        
                    st.success("Found matching background music!")
                    
                    track_names = [f"{result['name']} ({result['duration']:.1f}s)" 
                                 for result in music_results]
                    selected_track = st.selectbox("Select background music:", track_names)
                    
                    selected_idx = track_names.index(selected_track)
                    preview_url = music_results[selected_idx]['previews']['preview-hq-mp3']

                with st.spinner("Step 3: Processing and mixing audio..."):
                    background_music = download_and_process_audio(preview_url)
                    if not background_music:
                        st.error("Failed to process background music.")
                        return
                        
                    final_output = mix_audio(speech_file, background_music)
                    if not final_output:
                        st.error("Failed to mix audio.")
                        return

                    with open(final_output, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.success("Audio processing complete! Preview below:")
                        st.audio(audio_bytes, format="audio/wav")
                        
                        # Store activity in history
                        store_activity(
                            db,
                            st.session_state['username'],
                            "text_to_speech",
                            {
                                "text": text,
                                "voice": selected_display_name,
                                "background_music": selected_track
                            }
                        )
                        
                        if st.download_button(
                            "Download Audio",
                            audio_bytes,
                            f"speech_with_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                            mime="audio/wav"
                        ):
                            store_activity(
                                db,
                                st.session_state['username'],
                                "text_to_speech",
                                {
                                    "text": text,
                                    "voice": selected_display_name,
                                    "background_music": selected_track
                                }
                            )
                            st.success("Downloaded successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Text-to-speech error: {str(e)}")
                
            finally:
                # Cleanup temporary files
                try:
                    if os.path.exists("output.wav"):
                        os.remove("output.wav")
                    if os.path.exists("final_output.wav"):
                        os.remove("final_output.wav")
                except Exception as e:
                    logger.warning(f"Failed to cleanup output files: {str(e)}")
    else:
        st.error("Failed to fetch available voices")

def mix_audio(speech_path, background_music, output_path="final_output.wav"):
    """Mix speech with background music"""
    try:
        speech = AudioSegment.from_wav(speech_path)
        
        while len(background_music) < len(speech):
            background_music = background_music + background_music
        
        background_music = background_music[:len(speech)]
        final_audio = background_music.overlay(speech)
        
        final_audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        logger.error(f"Error mixing audio: {str(e)}")
        return None

def setup_ffmpeg():
    """Configure ffmpeg path for pydub"""
    try:
        AudioSegment.silent(duration=100)
        logger.debug("FFmpeg test successful")
        return True
    except Exception as e:
        logger.error(f"FFmpeg test failed: {str(e)}")
        return False

def search_background_music(query, freesound_api_key):
    """Search for background music using Freesound API"""
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

def download_and_process_audio(preview_url, volume_reduction=10):
    """Download and process background music"""
    temp_path = None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(preview_url, headers=headers)
        if response.status_code != 200:
            return None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
            
        try:
            audio = AudioSegment.from_mp3(temp_path)
            processed_audio = audio - volume_reduction
            return processed_audio
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error in download_and_process_audio: {str(e)}")
        return None
    
def file_upload_page(db):
    st.title("File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'doc', 'docx', 'jpg', 'png', 'mp3', 'wav']
    )
    
    if uploaded_file:
        # Get file details
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        st.json(file_details)
        
        # Store file metadata
        store_activity(
            db,
            st.session_state['username'],
            "file_upload",
            file_details
        )
        
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        
def main():
    st.set_page_config(page_title="AI Tools Hub", layout="wide")
    
    # Initialize database connection
    db = init_db()
    if not db:
        st.error("Failed to connect to database")
        return
    
    # Initialize session states
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Chatbot'
    
    create_header()
    
    if st.session_state.get('logged_in'):
        col1, col2 = st.columns([7, 3])
        
        with col1:
            if st.session_state['current_page'] == 'Chatbot':
                chatbot_page(db)
            elif st.session_state['current_page'] == 'Photo Editor':
                photo_editor_page(db)
            elif st.session_state['current_page'] == 'Text to Speech':
                text_to_speech_page(db)
            elif st.session_state['current_page'] == 'File Upload':
                file_upload_page(db)
        
        with col2:
            display_history(db, st.session_state['current_page'])
    else:
        if st.session_state['page'] == 'login':
            login_page(db)
        else:
            signup_page(db)

if __name__ == "__main__":
    main()