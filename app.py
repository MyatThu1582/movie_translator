
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import json
import subprocess
import uuid
import imageio_ffmpeg
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq API Client securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please configure it in your .env file."
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# Automatically use the bundled FFmpeg path (no Windows PATH needed!)
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# Load NLLB Model
MODEL_NAME = "facebook/nllb-200-distilled-600M"
print("Loading NLLB Translation Model into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

@app.get("/")
def read_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found in project folder."}

@app.post("/translate-video/")
async def translate_video(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())[:8]
        temp_video_path = f"temp_{session_id}_{file.filename}"
        
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())

        # 1. Extract audio using bundled FFmpeg path
        audio_path = f"temp_{session_id}.mp3"
        try:
            subprocess.run([
                FFMPEG_PATH, "-y", "-i", temp_video_path,
                "-vn", "-acodec", "libmp3lame", "-ab", "128k", audio_path
            ], check=True)
        except Exception as ffmpeg_err:
            return JSONResponse(status_code=500, content={"error": f"Audio extraction error: {str(ffmpeg_err)}"})

        # 2. Transcribe via Groq API
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(audio_path, audio_file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )
        except Exception as groq_err:
            return JSONResponse(status_code=500, content={"error": f"Groq API error: {str(groq_err)}"})

        burmese_subtitles = []

        # 3. Translate segments locally via NLLB
        for segment in transcription.segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            inputs = tokenizer(text, return_tensors="pt")
            translated_tokens = translation_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("mya_Mymr"),
                max_new_tokens=100
            )
            burmese_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            burmese_subtitles.append({
                "start": start_time,
                "end": end_time,
                "text": burmese_text
            })

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {
            "subtitles": burmese_subtitles,
            "video_filename": temp_video_path
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})

@app.post("/render-video/")
async def render_video(
    video_filename: str = Form(...),
    subtitles: str = Form(...)
):
    if not os.path.exists(video_filename):
        return JSONResponse(status_code=400, content={"error": "Original video file expired or not found."})

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

    # Use bundled FFMPEG_PATH here as well
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

    if os.path.exists(srt_path):
        os.remove(srt_path)

    return FileResponse(
        path=output_video,
        filename="translated_burmese_movie.mp4",
        media_type="video/mp4"
    )