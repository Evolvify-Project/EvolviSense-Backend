import logging
import cv2
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from collections import defaultdict
from moviepy import VideoFileClip
import tempfile
import subprocess
from pydub import AudioSegment
import ffmpeg
import threading
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a lock to prevent concurrent video processing
processing_lock = threading.Lock()

# Create FastAPI app
app = FastAPI(
    title="Emotion Analysis API",
    description="API for emotion analysis from video uploads with voice analysis."
)

# Mental health indicator weights for facial analysis
weights = {
    "depression": {"sad": 0.5, "fear": 0.2, "anger": 0.1, "disgust": 0.1, "neutral": 0.1},
    "stress": {"anger": 0.4, "fear": 0.3, "surprise": 0.1, "disgust": 0.1, "sad": 0.1},
    "anxiety": {"fear": 0.5, "surprise": 0.2, "sad": 0.2, "anger": 0.1}
}

# Weights for voice analysis
voice_weights = {
    "stress": {"pitch": 0.5, "energy": 0.5},
    "anxiety": {"pitch": 0.7, "energy": 0.3}
}

def normalize_score(score: float, max_score: float) -> float:
    return round(float(score) / max_score, 2) if max_score > 0 else 0

def convert_to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def compute_mental_health_scores(emotion_percentages: dict) -> dict:
    scores = {}
    joy_threshold = 0.5
    intensity_factor = 1.5
    is_positive_dominant = emotion_percentages.get("happy", 0) > joy_threshold
    total_negative_emotion = sum(emotion_percentages.get(e, 0) for e in ["sad", "fear", "anger", "disgust"])
    intensity_multiplier = intensity_factor if total_negative_emotion > 0.5 else 1.0

    for indicator, emotion_weights in weights.items():
        raw_score = sum(emotion_percentages.get(emotion, 0) * weight for emotion, weight in emotion_weights.items()
                        if not (is_positive_dominant and emotion in {"sad", "fear", "anger"}))
        max_score = sum(emotion_weights.values())
        normalized_score = min(1.0, normalize_score(raw_score * max_score, max_score))
        scores[indicator] = round(normalized_score * 100, 2)
    return convert_to_json_serializable(scores)

def detect_emotions(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(img_path=frame_rgb, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list) and result:
            result = result[0]
        elif not result:
            logger.info("No emotions detected, returning neutral")
            return {"emotion": "neutral", "confidence": 0.0}
        dominant_emotion = result["dominant_emotion"]
        confidence = result["emotion"][dominant_emotion]
        logger.info(f"Processed frame: {dominant_emotion} ({confidence:.2f})")
        return {"emotion": dominant_emotion, "confidence": float(confidence)}
    except ValueError as e:
        logger.warning(f"Face detection failed: {e}")
        return {"emotion": "neutral", "confidence": 0.0}
    except Exception as e:
        logger.error(f"Unexpected error in detect_emotions: {e}")
        return {"error": str(e), "emotion": "neutral", "confidence": 0.0}

def analyze_voice(audio_file: str) -> dict:
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) == 0:
            logger.info("No audio data found")
            return {"stress": 0, "anxiety": 0}

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean([p for p in pitches[pitches > 0]]) if np.any(pitches > 0) else 0
        energy = np.mean(librosa.feature.rms(y=y))
        pitch_normalized = min(float(pitch_mean) / 500, 1.0)
        energy_normalized = min(float(energy) / 0.1, 1.0)

        scores = {}
        for indicator, feature_weights in voice_weights.items():
            raw_score = (feature_weights["pitch"] * pitch_normalized + feature_weights["energy"] * energy_normalized)
            max_score = sum(feature_weights.values())
            scores[indicator] = round(normalize_score(raw_score, max_score) * 100, 2)

        logger.info(f"Voice analysis scores: {scores}")
        return convert_to_json_serializable(scores)
    except Exception as e:
        logger.error(f"Error analyzing voice: {e}")
        return {"stress": 0, "anxiety": 0}

def check_ffmpeg_installation():
    """Check if ffmpeg is installed and available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True, timeout=5)
        logger.info(f"FFmpeg is installed: {result.stdout.splitlines()[0]}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg is not installed or not accessible: {e}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg binary not found in system PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg check timed out")
        return False

def check_ffmpeg_codecs():
    """Check if FFmpeg supports necessary codecs for mp4 (H.264 and AAC)"""
    try:
        # Check for H.264 decoder (libx264)
        h264_result = subprocess.run(['ffmpeg', '-codecs'], capture_output=True, text=True, check=True, timeout=5)
        if 'libx264' not in h264_result.stdout and 'h264' not in h264_result.stdout:
            logger.error("FFmpeg does not support H.264 codec")
            return False
        
        # Check for AAC decoder
        aac_result = subprocess.run(['ffmpeg', '-codecs'], capture_output=True, text=True, check=True, timeout=5)
        if 'aac' not in aac_result.stdout:
            logger.error("FFmpeg does not support AAC codec")
            return False
        
        logger.info("FFmpeg supports H.264 and AAC codecs")
        return True
    except Exception as e:
        logger.error(f"Error checking FFmpeg codecs: {e}")
        return False

def extract_audio_with_ffmpeg(video_path: str, audio_path: str) -> bool:
    """Extract audio using ffmpeg"""
    try:
        logger.info(f"Ensuring output path for audio: {audio_path}")
        Path(audio_path).parent.mkdir(parents=True, exist_ok=True)
        
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ac', '2',  # Ensure 2 audio channels
            '-y', audio_path
        ]
        logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
        
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        logger.info(f"FFmpeg command output: {result.stdout}")
        logger.info(f"FFmpeg command stderr: {result.stderr}")
        logger.info(f"Audio extracted via ffmpeg to: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg audio extraction timed out")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during ffmpeg extraction: {e}")
        return False

def extract_audio_with_pydub(video_path: str, audio_path: str) -> bool:
    """Extract audio using pydub as a fallback"""
    try:
        logger.info(f"Attempting audio extraction with pydub: {video_path} -> {audio_path}")
        audio = AudioSegment.from_file(video_path, format="mp4")
        audio.export(audio_path, format="wav", codec="pcm_s16le")
        logger.info(f"Audio extracted via pydub to: {audio_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting audio with pydub: {e}")
        return False

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...), frame_skip: int = 10, max_frames: int = 50):
    logger.info(f"Received video upload: {file.filename}, size: {file.size}")
    if not processing_lock.acquire(blocking=False):
        logger.warning("Another video is being processed, rejecting new request")
        return JSONResponse(status_code=429, content={"error": "Please wait, processing previous video"})

    temp_video_path = None
    temp_audio_path = None
    try:
        contents = await file.read()
        if len(contents) > 100 * 1024 * 1024:
            logger.error("File size exceeds 100MB limit")
            return JSONResponse(status_code=413, content={"error": "File size exceeds 100MB limit"})

        # Save the uploaded file as mp4
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(contents)
            temp_video_path = temp_video.name
        logger.info(f"Temporary video saved at: {temp_video_path}")

        # Check FFmpeg installation and codec support
        if not check_ffmpeg_installation():
            logger.error("FFmpeg is not installed")
            return JSONResponse(status_code=500, content={"error": "FFmpeg is not installed on the server"})
        
        if not check_ffmpeg_codecs():
            logger.error("FFmpeg does not support required codecs for mp4")
            return JSONResponse(status_code=500, content={"error": "FFmpeg does not support H.264 or AAC codecs"})

        # Open video with moviepy
        try:
            video = VideoFileClip(temp_video_path)
            video_duration = video.duration
            logger.info(f"Video duration: {video_duration} seconds")
        except Exception as e:
            logger.error(f"Failed to open video with moviepy: {e}")
            return JSONResponse(status_code=400, content={"error": f"Invalid video: cannot be opened with MoviePy - {str(e)}"})
        finally:
            try:
                video.close()
            except:
                logger.warning("Failed to close video file with MoviePy")

        has_audio = False
        voice_scores = {"stress": 0, "anxiety": 0}
        try:
            with VideoFileClip(temp_video_path) as video:
                if video.audio:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                        temp_audio_path = temp_audio.name
                        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
                    logger.info(f"Audio extracted to: {temp_audio_path}")
                    voice_scores = analyze_voice(temp_audio_path)
                    has_audio = True
                else:
                    logger.warning("No audio track detected in video, proceeding with facial analysis only")
        except Exception as e:
            logger.warning(f"Error extracting audio with moviepy: {e}")
            if check_ffmpeg_installation():
                temp_audio_path = Path(tempfile.gettempdir()) / f"temp_audio_{os.urandom(8).hex()}.wav"
                if not extract_audio_with_ffmpeg(temp_video_path, str(temp_audio_path)) and not extract_audio_with_pydub(temp_video_path, str(temp_audio_path)):
                    logger.warning("Audio extraction failed with both methods, using default scores")
                else:
                    voice_scores = analyze_voice(str(temp_audio_path))
                    has_audio = True
            else:
                logger.warning("FFmpeg not installed, skipping audio extraction fallback")

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error("Could not open video file with OpenCV")
            return JSONResponse(status_code=400, content={"error": "Invalid video: cannot be opened with OpenCV"})

        emotion_history = defaultdict(list)
        frame_data = []
        frame_count = 0
        processed_frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            emotion_result = detect_emotions(frame)
            if "error" in emotion_result:
                logger.error(f"Emotion detection error: {emotion_result['error']}")
                cap.release()
                return JSONResponse(status_code=500, content=emotion_result)

            emotion = emotion_result["emotion"]
            confidence = emotion_result["confidence"]
            emotion_history[emotion].append(confidence)
            processed_frames += 1
            logger.info(f"Processed frame {processed_frames}: {emotion} ({confidence:.2f})")

            frame_emotion_percentages = {k: 1 if k == emotion else 0 for k in weights["depression"].keys()}
            frame_facial_scores = compute_mental_health_scores(frame_emotion_percentages)
            frame_data.append({
                "time": round(frame_count / fps, 2),
                "stress": (0.7 * frame_facial_scores["stress"] + 0.3 * voice_scores["stress"]) if has_audio else frame_facial_scores["stress"],
                "anxiety": (0.7 * frame_facial_scores["anxiety"] + 0.3 * voice_scores["anxiety"]) if has_audio else frame_facial_scores["anxiety"],
                "confidence": confidence
            })

        cap.release()
        logger.info("Video processing completed")

        if processed_frames > 0:
            emotion_percentages = {k: len(v) / processed_frames for k, v in emotion_history.items() if v}
            facial_scores = compute_mental_health_scores(emotion_percentages)
        else:
            logger.info("No frames processed, returning default scores")
            facial_scores = {"depression": 0, "stress": 0, "anxiety": 0}

        final_scores = {
            "depression": facial_scores["depression"],
            "stress": (0.7 * facial_scores["stress"] + 0.3 * voice_scores["stress"]) if has_audio else facial_scores["stress"],
            "anxiety": (0.7 * facial_scores["anxiety"] + 0.3 * voice_scores["anxiety"]) if has_audio else facial_scores["anxiety"],
            "confidence": sum(d["confidence"] for d in frame_data) / len(frame_data) if frame_data else 0.0
        }

        peak_stress = max(d["stress"] for d in frame_data) if frame_data else 0.0

        logger.info(f"Final scores: {final_scores}")
        response_data = {
            "emotions": convert_to_json_serializable(dict(emotion_history)),
            "mental_health": convert_to_json_serializable(final_scores),
            "voice_analysis": convert_to_json_serializable(voice_scores) if has_audio else None,
            "frame_data": convert_to_json_serializable(frame_data),
            "peak_stress": peak_stress,
            "video_duration": video_duration
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
    finally:
        if temp_video_path and Path(temp_video_path).exists():
            logger.info(f"Cleaning up temporary video file: {temp_video_path}")
            Path(temp_video_path).unlink(missing_ok=True)
        if temp_audio_path and Path(temp_audio_path).exists():
            logger.info(f"Cleaning up temporary audio file: {temp_audio_path}")
            Path(temp_audio_path).unlink(missing_ok=True)
        processing_lock.release()