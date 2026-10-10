
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq


# ==================================================
# 1. CONFIGURATION
# ==================================================

load_dotenv()

app = FastAPI(title="AI Movie Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Configure it in your .env file."
    )

groq_client = Groq(api_key=GROQ_API_KEY)
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_files"
TEMP_DIR.mkdir(exist_ok=True)

# Use the model that worked in your previous test.
TRANSLATION_MODEL = "openai/gpt-oss-20b"

# Translate a few subtitle segments per request.
# Smaller batches reduce output-length problems.
BATCH_SIZE = 8

print("AI MOVIE TRANSLATOR BACKEND READY!")
print("URL: http://127.0.0.1:8000/")


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def format_time(seconds):
    """Convert seconds to SRT timestamp format."""

    total_ms = round(float(seconds) * 1000)

    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    milliseconds = total_ms % 1000

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{secs:02d},{milliseconds:03d}"
    )


def contains_myanmar(text):
    """Check whether text contains Myanmar Unicode characters."""

    return bool(
        re.search(r"[\u1000-\u109F\uAA60-\uAA7F]", text)
    )


def build_translation_prompt(segments):
    """Create a translation prompt for one batch."""

    numbered_segments = [
        {
            "id": index,
            "text": segment["text"],
        }
        for index, segment in enumerate(segments)
    ]

    return f"""
You are a professional movie subtitle translator fluent in Burmese.

Translate every dialogue segment below into natural, conversational,
emotionally appropriate Burmese (Myanmar).

Requirements:
1. Translate the meaning of every dialogue into Burmese.
2. Preserve the original meaning, emotion, and tone.
3. Do not summarize, merge, omit, or reorder segments.
4. Keep each segment ID exactly as provided.
5. Keep timestamps unchanged; timestamps are handled separately.
6. Names may remain unchanged when appropriate.
7. Short expressions should still be translated naturally.
8. Return a JSON object in exactly this structure:

{{
  "translations": [
    {{"id": 0, "text": "Burmese translation"}}
  ]
}}

Return one translation for every supplied ID.
Do not include Markdown code fences or explanations.

Dialogue segments:
{json.dumps(numbered_segments, ensure_ascii=False)}
""".strip()


def parse_translation_response(response_text, expected_count):
    """Parse JSON and verify that all requested segment IDs exist."""

    if not response_text or not response_text.strip():
        raise ValueError("The model returned an empty response.")

    cleaned = response_text.strip()

    # Remove Markdown code fences if present.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Find JSON if extra text surrounds it.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "The model did not return a valid JSON object."
        )

    data = json.loads(cleaned[start:end + 1])

    if isinstance(data, dict):
        translations_data = data.get("translations", [])
    elif isinstance(data, list):
        # Accept a plain list as a fallback format.
        translations_data = data
    else:
        raise ValueError("Unexpected translation response format.")

    if not isinstance(translations_data, list):
        raise ValueError("'translations' must be a JSON array.")

    translations_by_id = {}

    for item in translations_data:
        if not isinstance(item, dict):
            continue

        segment_id = item.get("id")
        translated_text = item.get("text")

        # JSON IDs should be integers, not booleans.
        if isinstance(segment_id, bool):
            continue

        if not isinstance(segment_id, int):
            continue

        if isinstance(translated_text, str) and translated_text.strip():
            translations_by_id[segment_id] = translated_text.strip()

    expected_ids = set(range(expected_count))
    actual_ids = set(translations_by_id.keys())

    if expected_ids != actual_ids:
        missing = sorted(expected_ids - actual_ids)
        raise ValueError(
            f"Missing translations for segment IDs: {missing}"
        )

    return [
        translations_by_id[index]
        for index in range(expected_count)
    ]



def translate_batch(segments, batch_number):
    """Translate a subtitle batch into natural, human-style Burmese."""

    prompt = build_translation_prompt(segments)
    last_error = None

    system_prompt = """
You are an experienced native Burmese (Myanmar) movie subtitle translator.

Your job is to make movie dialogue sound natural, emotional, and human,
as if a skilled Burmese translator localized the movie for local viewers.

TRANSLATION STYLE:
1. Use everyday, conversational Burmese rather than formal written Burmese.
2. Make dialogue sound like something a real person would say aloud.
3. Match each character's personality, age, relationship, and emotion.
4. Use natural expressions for romance, anger, sadness, jokes, teasing,
   sarcasm, surprise, fear, and arguments.
5. Avoid stiff, bookish, robotic, overly polite, or literal translations
   unless the character is intentionally speaking formally.
6. Do not translate word-for-word if that sounds unnatural in Burmese.
   Preserve the meaning using a natural Burmese expression instead.
7. Keep subtitles concise and easy to read while watching the movie.
8. Preserve emotional intensity. Do not make an angry character sound polite
   or a casual character sound like they are giving a formal speech.
9. Use pronouns and forms of address that fit the relationship and context.
   Do not automatically use the same pronouns for every character.
10. Preserve jokes, implied meaning, and conversational intent.
11. Do not add explanations, extra dialogue, or information absent from
    the original.
12. Keep names and unavoidable foreign terms where appropriate.
13. Translate every segment. Never omit, merge, or reorder segments.
14. Preserve every segment ID exactly as supplied.
15. Return ONLY a valid JSON object in the requested format.

IMPORTANT:
Natural dialogue is more important than literal word-for-word translation.
The result should feel like real Burmese movie subtitles, not a textbook,
news report, official document, or AI-generated explanation.
"""

    # Retry once if the model returns malformed or incomplete JSON.
    for attempt in range(1, 3):
        try:
            print(
                f"[DEBUG] Batch {batch_number}, attempt {attempt}: "
                "requesting natural Burmese translation..."
            )

            completion = groq_client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_completion_tokens=4096,
            )

            choice = completion.choices[0]
            response_text = choice.message.content or ""

            print(
                f"[DEBUG] Batch {batch_number} finish reason: "
                f"{choice.finish_reason}"
            )

            print(
                f"[DEBUG] Batch {batch_number} raw response: "
                f"{response_text[:1500]!r}"
            )

            if not response_text.strip():
                raise ValueError("The model returned an empty response.")

            translations = parse_translation_response(
                response_text,
                expected_count=len(segments),
            )

            for index, translated_text in enumerate(translations, start=1):
                print(
                    f"[DEBUG] Batch {batch_number}, "
                    f"segment {index}: {translated_text}"
                )

            # Warn about output that may not be Burmese.
            non_burmese = [
                index
                for index, text in enumerate(translations)
                if not contains_myanmar(text)
            ]

            if non_burmese:
                print(
                    f"[WARNING] Batch {batch_number}: translations "
                    f"without Myanmar characters at positions "
                    f"{non_burmese}"
                )

            return translations

        except Exception as error:
            last_error = error

            print(
                f"[WARNING] Batch {batch_number}, attempt {attempt} "
                f"failed: {error}"
            )

    raise RuntimeError(
        f"Translation failed for batch {batch_number}: {last_error}"
    )


    prompt = build_translation_prompt(segments)
    last_error = None

    # Two attempts help with occasional malformed responses.
    for attempt in range(1, 3):
        try:
            print(
                f"[DEBUG] Batch {batch_number}, "
                f"attempt {attempt}: requesting translation..."
            )

            completion = groq_client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate movie dialogue into natural Burmese. "
                            "Return only the requested JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_completion_tokens=4096,
            )

            choice = completion.choices[0]
            response_text = choice.message.content or ""

            print(
                f"[DEBUG] Batch {batch_number} finish reason: "
                f"{choice.finish_reason}"
            )
            print(
                f"[DEBUG] Batch {batch_number} raw response: "
                f"{response_text[:1500]!r}"
            )

            if not response_text.strip():
                raise ValueError(
                    "The model returned an empty response."
                )

            translations = parse_translation_response(
                response_text,
                expected_count=len(segments),
            )

            # Log Burmese output for debugging.
            for index, translated_text in enumerate(translations, start=1):
                print(
                    f"[DEBUG] Batch {batch_number}, "
                    f"segment {index}: {translated_text}"
                )

            # Warn about possible untranslated text without rejecting
            # proper names or short expressions automatically.
            non_burmese = [
                index
                for index, text in enumerate(translations)
                if not contains_myanmar(text)
            ]

            if non_burmese:
                print(
                    "[WARNING] These translations contain no Myanmar "
                    f"characters: {non_burmese}"
                )

            return translations

        except Exception as error:
            last_error = error

            print(
                f"[WARNING] Batch {batch_number}, attempt {attempt} "
                f"failed: {error}"
            )

    raise RuntimeError(
        f"Translation failed for batch {batch_number}: {last_error}"
    )


def write_srt(subtitles, srt_path):
    """Write subtitle segments to an SRT file."""

    with open(
        srt_path,
        "w",
        encoding="utf-8-sig",
        newline="\n",
    ) as file:
        for index, subtitle in enumerate(subtitles, start=1):
            start = format_time(subtitle["start"])
            end = format_time(subtitle["end"])
            text = str(subtitle["text"]).strip()

            file.write(
                f"{index}\n"
                f"{start} --> {end}\n"
                f"{text}\n\n"
            )


def escape_subtitle_path(path):
    """Escape a Windows or Unix path for FFmpeg's subtitles filter."""

    path = Path(path).resolve().as_posix()

    # Escape characters interpreted specially by FFmpeg.
    for char in ("\\", ":", "'", ",", "[", "]"):
        path = path.replace(char, "\\" + char)

    return path


# ==================================================
# 3. HOME PAGE
# ==================================================

@app.get("/")
def read_index():
    index_path = BASE_DIR / "index.html"

    if index_path.is_file():
        return FileResponse(index_path)

    return JSONResponse(
        status_code=404,
        content={
            "error": "index.html not found in the project folder."
        },
    )


# ==================================================
# 4. TRANSCRIBE AND TRANSLATE VIDEO
# ==================================================

@app.post("/translate-video/")
async def translate_video(file: UploadFile = File(...)):
    session_id = uuid.uuid4().hex[:10]

    video_path = TEMP_DIR / f"video_{session_id}.mp4"
    audio_path = TEMP_DIR / f"audio_{session_id}.mp3"

    try:
        # ------------------------------------------
        # Save uploaded video
        # ------------------------------------------

        print(f"\n[DEBUG 1] Uploaded file: {file.filename}")

        with open(video_path, "wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                output_file.write(chunk)

        await file.close()

        if not video_path.is_file() or video_path.stat().st_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "The uploaded video is empty."},
            )

        # ------------------------------------------
        # Extract audio
        # ------------------------------------------

        print("[DEBUG 1A] Extracting audio...")

        subprocess.run(
            [
                FFMPEG_PATH,
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "libmp3lame",
                "-ab", "128k",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # ------------------------------------------
        # Transcribe audio
        # ------------------------------------------

        print("[DEBUG 2] Transcribing audio via Groq Whisper...")

        with open(audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        raw_segments = []

        for segment in transcription.segments or []:
            if isinstance(segment, dict):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = str(segment.get("text", "")).strip()
            else:
                start = segment.start
                end = segment.end
                text = str(segment.text).strip()

            if text:
                raw_segments.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "text": text,
                    }
                )

        if not raw_segments:
            return JSONResponse(
                status_code=422,
                content={
                    "error": (
                        "No speech segments were detected in this video."
                    )
                },
            )

        print(f"[DEBUG 3] Found {len(raw_segments)} segments.")
        print(
            f"[DEBUG 3A] Translating in batches of {BATCH_SIZE}..."
        )

        # Keep a copy of the original segments for debugging,
        # but never return them as successful translations.
        original_segments = [
            dict(segment) for segment in raw_segments
        ]

        # ------------------------------------------
        # Translate in batches
        # ------------------------------------------

        translated_segments = []

        total_batches = (
            len(raw_segments) + BATCH_SIZE - 1
        ) // BATCH_SIZE

        for batch_index, start_index in enumerate(
            range(0, len(raw_segments), BATCH_SIZE),
            start=1,
        ):
            batch = raw_segments[
                start_index:start_index + BATCH_SIZE
            ]

            print(
                f"[DEBUG 3B] Processing batch {batch_index}/"
                f"{total_batches}, containing {len(batch)} segments."
            )

            translations = translate_batch(batch, batch_index)

            for original_segment, translated_text in zip(
                batch,
                translations,
            ):
                translated_segments.append(
                    {
                        "start": original_segment["start"],
                        "end": original_segment["end"],
                        "text": translated_text,
                    }
                )

        if len(translated_segments) != len(original_segments):
            raise RuntimeError(
                "The translated segment count does not match "
                "the transcription segment count."
            )

        print(
            f"[DEBUG 4] Successfully translated "
            f"{len(translated_segments)} segments into Burmese."
        )

        return {
            "subtitles": translated_segments,
            "video_filename": video_path.name,
        }

    except subprocess.CalledProcessError as error:
        details = error.stderr[-2000:] if error.stderr else str(error)

        print(f"[ERROR] FFmpeg failed: {details}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to extract audio from the video.",
                "details": details,
            },
        )

    except Exception as error:
        print(f"[ERROR] Video translation failed: {error}")

        return JSONResponse(
            status_code=502,
            content={
                "error": (
                    "The video was transcribed, but Burmese translation "
                    "failed. Please try again with a shorter video."
                ),
                "details": str(error),
            },
        )

    finally:
        # The original video is intentionally kept because the
        # render endpoint needs it after translation completes.
        if audio_path.exists():
            try:
                audio_path.unlink()
            except OSError:
                pass


# ==================================================
# 5. RENDER VIDEO WITH BURMESE SUBTITLES
# ==================================================

@app.post("/render-video/")
async def render_video(
    video_filename: str = Form(...),
    subtitles: str = Form(...),
):
    session_id = uuid.uuid4().hex[:10]

    srt_path = TEMP_DIR / f"subtitles_{session_id}.srt"
    output_path = TEMP_DIR / f"output_{session_id}.mp4"

    try:
        # ------------------------------------------
        # Validate video filename
        # ------------------------------------------

        if (
            Path(video_filename).name != video_filename
            or not video_filename.startswith("video_")
            or not video_filename.endswith(".mp4")
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid video filename."},
            )

        video_path = TEMP_DIR / video_filename

        if not video_path.is_file():
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Original video file not found. "
                        "Please upload the video again."
                    )
                },
            )

        # ------------------------------------------
        # Parse subtitles
        # ------------------------------------------

        try:
            subtitle_segments = json.loads(subtitles)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid subtitle JSON."},
            )

        if not isinstance(subtitle_segments, list) or not subtitle_segments:
            return JSONResponse(
                status_code=400,
                content={"error": "No subtitle segments were provided."},
            )

        for segment in subtitle_segments:
            if not isinstance(segment, dict):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid subtitle segment."},
                )

            if not all(
                key in segment for key in ("start", "end", "text")
            ):
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "A subtitle segment is missing fields."
                    },
                )

            if not isinstance(segment["text"], str):
                return JSONResponse(
                    status_code=400,
                    content={"error": "Subtitle text must be a string."},
                )

        # ------------------------------------------
        # Generate SRT file
        # ------------------------------------------

        write_srt(subtitle_segments, srt_path)

        escaped_srt = escape_subtitle_path(srt_path)

        # ------------------------------------------
        # Render with FFmpeg
        # ------------------------------------------

        ffmpeg_command = [
            FFMPEG_PATH,
            "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{escaped_srt}'",
            "-c:a", "copy",
            str(output_path),
        ]

        print("[DEBUG 5] Rendering video with Burmese subtitles...")

        subprocess.run(
            ffmpeg_command,
            check=True,
            capture_output=True,
            text=True,
        )

        if not output_path.is_file():
            raise RuntimeError("FFmpeg did not create the output video.")

        print("[DEBUG 6] Video rendering completed.")

        return FileResponse(
            path=output_path,
            filename="translated_burmese_movie.mp4",
            media_type="video/mp4",
        )

    except subprocess.CalledProcessError as error:
        details = error.stderr[-3000:] if error.stderr else str(error)

        print(f"[ERROR] Video rendering failed: {details}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Video rendering failed.",
                "details": details,
            },
        )

    except Exception as error:
        print(f"[ERROR] Render server error: {error}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred while rendering the video.",
                "details": str(error),
            },
        )

    finally:
        if srt_path.exists():
            try:
                srt_path.unlink()
            except OSError:
                pass


# ==================================================
# 6. START SERVER
# ==================================================
# Run in the terminal:
# python -m uvicorn app:app --reload
