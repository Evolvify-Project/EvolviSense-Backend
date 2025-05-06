import logging
import cv2
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from collections import defaultdict
import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import os
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        normalized_score = min(1.0, normalize_score(raw_score * intensity_multiplier, max_score))
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

@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...), frame_skip: int = 10, max_frames: int = 50):
    logger.info(f"Received video upload: {file.filename}, size: {file.size}")
    try:
        # Check file size
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            logger.error("File size exceeds 10MB limit")
            return JSONResponse(status_code=413, content={"error": "File size exceeds 10MB limit"})

        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_video:
            temp_video.write(contents)
            temp_video_path = temp_video.name
        logger.info(f"Temporary video saved at: {temp_video_path}")

        # Extract audio
        has_audio = False
        voice_scores = {"stress": 0, "anxiety": 0}
        try:
            with moviepy.VideoFileClip(temp_video_path) as video:
                if video.audio:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                        video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
                        temp_audio_path = temp_audio.name
                    logger.info(f"Audio extracted to: {temp_audio_path}")
                    voice_scores = analyze_voice(temp_audio_path)
                    has_audio = True
                    os.unlink(temp_audio_path)
                else:
                    logger.info("No audio found in video")
        except Exception as e:
            logger.warning(f"No audio or error extracting audio: {e}")
            # Fallback using ffmpeg directly if moviepy fails
            try:
                temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                subprocess.run(['ffmpeg', '-i', temp_video_path, '-vn', '-acodec', 'pcm_s16le', temp_audio_path, '-y'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(f"Audio extracted via ffmpeg to: {temp_audio_path}")
                voice_scores = analyze_voice(temp_audio_path)
                has_audio = True
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg extraction failed: {e}")
            finally:
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

        # Process video for facial analysis
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error("Could not open video file")
            os.unlink(temp_video_path)
            return JSONResponse(status_code=400, content={"error": "Could not open video file"})

        emotion_history = defaultdict(list)
        frame_count = 0
        processed_frames = 0

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
                os.unlink(temp_video_path)
                return JSONResponse(status_code=500, content=emotion_result)

            emotion = emotion_result["emotion"]
            confidence = emotion_result["confidence"]
            emotion_history[emotion].append(confidence)
            processed_frames += 1
            logger.info(f"Processed frame {processed_frames}: {emotion} ({confidence:.2f})")

        cap.release()
        os.unlink(temp_video_path)
        logger.info("Video processing completed, temporary file deleted")

        # Compute facial mental health scores
        if processed_frames > 0:
            emotion_percentages = {k: len(v) / processed_frames for k, v in emotion_history.items() if v}
            facial_scores = compute_mental_health_scores(emotion_percentages)
        else:
            logger.info("No frames processed, returning default scores")
            facial_scores = {"depression": 0, "stress": 0, "anxiety": 0}

        # Combine facial and voice scores
        final_scores = {
            "depression": facial_scores["depression"],
            "stress": (0.7 * facial_scores["stress"] + 0.3 * voice_scores["stress"]) if has_audio else facial_scores["stress"],
            "anxiety": (0.7 * facial_scores["anxiety"] + 0.3 * voice_scores["anxiety"]) if has_audio else facial_scores["anxiety"],
            "confidence": facial_scores.get("confidence", 0)
        }

        logger.info(f"Final scores: {final_scores}")
        response_data = {
            "emotions": convert_to_json_serializable(dict(emotion_history)),
            "mental_health": convert_to_json_serializable(final_scores),
            "voice_analysis": convert_to_json_serializable(voice_scores) if has_audio else None
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
    finally:
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            logger.info(f"Cleaning up temporary file: {temp_video_path}")
            os.unlink(temp_video_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)