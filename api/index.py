import os
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

# --- HUGGING FACE SPACES ---
SPACE_TEXT_EMO  = "E-motionAssistant/Space4"
SPACE_AUDIO_EMO = "E-motionAssistant/Space5"
SPACE_LLM       = "E-motionAssistant/TherapyTamil"
SPACE_TTS       = "E-motionAssistant/Space3"

# --- IN-MEMORY STORE (Note: Clears on Vercel restart) ---
users_db: dict[str, dict] = {}
chat_sessions: dict[str, list] = {}

# ── REQUEST MODELS ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str
    message: str
    language: str = "tamil"
    type: str = "text"

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

# ── SERVE HTML FRONTEND ──────────────────────────────────────────────────────

# This is what fixes the {"detail":"Not Found"} error when visiting the main URL
@app.get("/")
async def serve_home():
    # Looks for public/chat.html one level up from the api folder
    static_file_path = os.path.join(os.path.dirname(__file__), "..", "public", "chat.html")
    if os.path.exists(static_file_path):
        return FileResponse(static_file_path)
    return {"error": "chat.html not found in public folder", "path_searched": static_file_path}

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(body: SignupRequest):
    email = body.email.strip().lower()
    if email in users_db:
        return {"success": False, "error": "An account with that email already exists."}
    users_db[email] = {"name": body.name, "email": email, "password": body.password}
    return {"success": True, "user": {"name": body.name, "email": email}}

@app.post("/auth/login")
def login(body: LoginRequest):
    identifier = body.username.strip().lower()
    user = users_db.get(identifier)
    if not user or user["password"] != body.password:
        return {"success": False, "error": "Invalid credentials."}
    return {"success": True, "user": {"name": user["name"], "email": user["email"]}}

# ── MAIN AI PIPELINE ──────────────────────────────────────────────────────────

@app.post("/api/python/predict")
def unified_ai_pipeline(body: PredictRequest):
    try:
        user_input = body.message
        lang       = body.language.lower()
        mode       = body.type
        sid        = body.session_id

        # 1. RETRIEVE MEMORY
        if sid not in chat_sessions:
            chat_sessions[sid] = []
        
        history = chat_sessions[sid][-10:]
        context_string = ""
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            context_string += f"{role_label}: {msg['content']}\n"

        # 2. DETECT EMOTION
        client_emo = Client(SPACE_TEXT_EMO if mode == "text" else SPACE_AUDIO_EMO, hf_token=HF_TOKEN)
        emotion_result = client_emo.predict(user_input, api_name="/predict")
        final_emotion = emotion_result if isinstance(emotion_result, str) else emotion_result.get("label", "neutral")

        # 3. GENERATE LLM RESPONSE
        client_llm = Client(SPACE_LLM, hf_token=HF_TOKEN)
        client_llm.timeout = 360
        
        full_prompt = (
            f"You are a helpful assistant. Language: {lang}. Detected Emotion: {final_emotion}.\n"
            f"Previous Conversation:\n{context_string}"
            f"User: {user_input}\n"
            f"Assistant:"
        )
        
        llm_reply = client_llm.predict(full_prompt, api_name="/chat")

        # 4. UPDATE MEMORY
        chat_sessions[sid].append({"role": "user", "content": user_input})
        chat_sessions[sid].append({"role": "assistant", "content": llm_reply})

        # 5. TEXT-TO-SPEECH
        client_tts = Client(SPACE_TTS, hf_token=HF_TOKEN)
        temp_audio_path = client_tts.predict(llm_reply, lang, api_name=f"/{lang}_tts")

        # 6. ENCODE AUDIO
        audio_base64 = ""
        if temp_audio_path and os.path.exists(temp_audio_path):
            with open(temp_audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "emotion": final_emotion,
            "reply_text": llm_reply,
            "reply_audio_base64": f"data:audio/wav;base64,{audio_base64}",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}