import os
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

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
SPACE_LLM       = "E-motionAssistant/TherapyEnglish"
SPACE_TTS       = "E-motionAssistant/Space3"

# --- In-memory user store (replace with a real DB for production) ---
users_db: dict[str, dict] = {}


# ── REQUEST MODELS ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    message: str
    language: str = "english"
    type: str = "text"   # "text" or "voice"

class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


# ── HEALTH CHECK ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Emotion Assistant API",
        "status": "running",
        "endpoints": {
            "health": "/api/python",
            "predict": "/api/python/predict",
            "signup": "/auth/signup",
            "login": "/auth/login"
        }
    }

@app.get("/api/python")
def health_check():
    return {"status": "FastAPI server is running"}


# ── AUTH ROUTES ───────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(body: SignupRequest):
    email = body.email.strip().lower()
    if email in users_db:
        return {"success": False, "error": "An account with that email already exists."}
    users_db[email] = {"name": body.name, "email": email, "password": body.password}
    return {
        "success": True,
        "user": {"name": body.name, "email": email},
    }

@app.post("/auth/login")
def login(body: LoginRequest):
    # Accept email OR username (email used as key)
    identifier = body.username.strip().lower()
    user = users_db.get(identifier)
    if not user or user["password"] != body.password:
        return {"success": False, "error": "Invalid credentials."}
    return {
        "success": True,
        "user": {"name": user["name"], "email": user["email"]},
    }


# ── MAIN AI PIPELINE ──────────────────────────────────────────────────────────

@app.post("/api/python/predict")
def unified_ai_pipeline(body: PredictRequest):
    try:
        user_input = body.message
        lang       = body.language.lower()
        mode       = body.type

        # STEP 1 — Detect emotion
        if mode == "text":
            client_emo    = Client(SPACE_TEXT_EMO, hf_token=HF_TOKEN)
            emotion_result = client_emo.predict(user_input, api_name="/predict")
        else:
            client_emo    = Client(SPACE_AUDIO_EMO, hf_token=HF_TOKEN)
            emotion_result = client_emo.predict(user_input, api_name="/predict")

        final_emotion = (
            emotion_result
            if isinstance(emotion_result, str)
            else emotion_result.get("label", "neutral")
        )

        # STEP 2 — Generate LLM response
        client_llm         = Client(SPACE_LLM, hf_token=HF_TOKEN)
        client_llm.timeout = 360
        prompt             = f"Language: {lang}. Detected Emotion: {final_emotion}. User said: {user_input}"
        llm_reply          = client_llm.predict(prompt, api_name="/chat")

        # STEP 3 — Text-to-speech
        client_tts     = Client(SPACE_TTS, hf_token=HF_TOKEN)
        temp_audio_path = client_tts.predict(llm_reply, lang, api_name=f"/{lang}_tts")

        # STEP 4 — Encode audio as base64
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