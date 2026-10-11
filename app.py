from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from openai import OpenAI
import os
import json
import subprocess
import uuid
import datetime
import imageio_ffmpeg
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in your .env file.")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing in your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# --- OPENROUTER DAILY USAGE TRACKER SETUP ---
QUOTA_FILE = "openrouter_usage.json"
MAX_DAILY_QUOTA = 50  # OpenRouter free tier limit guideline

def check_openrouter_quota():
    today_str = datetime.date.today().isoformat()
    data = {"date": today_str, "used": 0}
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r") as f:
                saved = json.load(f)
                if saved.get("date") == today_str:
                    data = saved
        except Exception:
            pass
    return MAX_DAILY_QUOTA - data["used"]

def increment_openrouter_quota():
    today_str = datetime.date.today().isoformat()
    data = {"date": today_str, "used": 0}
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r") as f:
                saved = json.load(f)
                if saved.get("date") == today_str:
                    data = saved
        except Exception:
            pass
    data["date"] = today_str
    data["used"] += 1
    with open(QUOTA_FILE, "w") as f:
        json.dump(data, f)
    return MAX_DAILY_QUOTA - data["used"]

print("🚀 ZERO-STORAGE SUBTITLE ENGINE READY WITH QUOTA TRACKER!")

@app.get("/")
def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found in project folder."}

@app.get("/quota-status/")
def quota_status():
    remaining = check_openrouter_quota()
    return {
        "daily_limit": MAX_DAILY_QUOTA,
        "remaining_requests": remaining,
        "reset_period": "Daily at midnight"
    }

@app.post("/translate-video/")
async def translate_video(file: UploadFile = File(...)):
    # 1. Check OpenRouter daily quota before starting
    remaining = check_openrouter_quota()
    if remaining <= 0:
        return JSONResponse(
            status_code=429, 
            content={"error": "OpenRouter free daily limit (50 requests) reached for today. Please try again tomorrow!"}
        )

    session_id = str(uuid.uuid4())[:8]
    temp_video_path = f"temp_{session_id}_{file.filename}"
    audio_path = f"temp_{session_id}.mp3"
    
    try:
        # Save uploaded file temporarily for processing (deleted immediately after)
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())

        # 2. Extract audio via FFmpeg
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", temp_video_path,
            "-vn", "-acodec", "libmp3lame", "-ab", "128k", audio_path
        ], check=True)

        # 3. Transcribe via Groq Whisper
        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        raw_segments = []
        for segment in transcription.segments:
            text = segment['text'].strip()
            if text:
                raw_segments.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": text
                })

        # 4. Translate lines into natural Burmese via DeepSeek on OpenRouter in chunks
        if raw_segments:
            CHUNK_SIZE = 100
            for i in range(0, len(raw_segments), CHUNK_SIZE):
                chunk = raw_segments[i:i + CHUNK_SIZE]
                texts_to_translate = [s["text"] for s in chunk]
                joined_text = " ||| ".join(texts_to_translate)
                
                prompt = (
                    "You are an expert native movie subtitle translator. Translate the following dialogue lines "
                    "from any foreign language into natural, conversational, and emotional Burmese (Myanmar) movie subtitles. "
                    "Keep the ' ||| ' separator exactly as it is between each translated sentence. Do not add or remove separators. "
                    "Output ONLY the translated Burmese text with the separators.\n\n" + joined_text
                )

                response = openrouter_client.chat.completions.create(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                translated_blob = response.choices[0].message.content.strip()
                translated_list = [t.strip() for t in translated_blob.split("|||")]

                for idx, seg in enumerate(chunk):
                    if idx < len(translated_list) and translated_list[idx]:
                        seg["text"] = translated_list[idx]

            # Successfully completed translation; deduct from quota
            increment_openrouter_quota()

        return {
            "subtitles": raw_segments,
            "video_filename": temp_video_path
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})
    finally:
        # Cleanup temporary audio file immediately
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/render-video/")
async def render_video(
    video_filename: str = Form(...),
    subtitles: str = Form(...)
):
    if not os.path.exists(video_filename):
        return JSONResponse(status_code=400, content={"error": "Original video file expired or not found. Please re-upload."})

    subs = json.loads(subtitles)
    session_id = str(uuid.uuid4())[:8]
    srt_path = f"subs_{session_id}.srt"
    output_video = f"output_{session_id}.mp4"

    def format_time(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millisecs:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sub in enumerate(subs, 1):
            start = format_time(sub['start'])
            end = format_time(sub['end'])
            f.write(f"{idx}\n{start} --> {end}\n{sub['text']}\n\n")

    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")

    ffmpeg_cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_filename,
        "-vf", f"subtitles='{escaped_srt}'",
        "-c:a", "copy",
        output_video
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Video rendering failed: {str(e)}"})
    finally:
        if os.path.exists(srt_path):
            os.remove(srt_path)
        # Clean up original temporary uploaded video file right after rendering is complete
        if os.path.exists(video_filename):
            os.remove(video_filename)

    return FileResponse(
        path=output_video,
        filename="translated_burmese_movie.mp4",
        media_type="video/mp4"
    )