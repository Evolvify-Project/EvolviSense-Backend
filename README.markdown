---
title: EvolviSense
emoji: 🏢
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: An API that delivers emotion analysis using DeepFace
---

# Emotion Analysis API

This Hugging Face Space hosts a **FastAPI** application that performs emotion analysis from uploaded video files. It leverages the `DeepFace` library for facial emotion detection (e.g., happy, sad, fear) and `Librosa` for voice analysis to compute mental health scores (depression, stress, anxiety). The API processes video inputs to provide detailed emotion summaries and mental health insights.

## Features
- **Uploaded Video Analysis**: Detects emotions from video files (e.g., WebM, MP4) with a maximum size of 10MB.
- **Facial Emotion Detection**: Uses `DeepFace` to identify emotions in video frames with confidence scores.
- **Voice Analysis**: Analyzes audio extracted from videos to compute stress and anxiety scores based on pitch and energy.
- **Mental Health Scores**: Calculates scores for depression, stress, and anxiety based on facial and voice analysis.
- **Robust Error Handling**: Includes comprehensive logging and error handling for reliable operation.

## How to Use

### Uploaded Video Analysis (HTTP POST)
- **Endpoint**: `POST /analyze-video/`
- **Description**: Upload a video file to analyze emotions, mental health scores, and voice features.
- **Parameters**:
  - `file` (required): Video file (e.g., WebM, MP4; max 10MB).
  - `frame_skip` (optional): Number of frames to skip during analysis (default: 10).
  - `max_frames` (optional): Maximum number of frames to analyze (default: 50).
- **Response**:
  - `emotions`: Dictionary of detected emotions with confidence scores for each frame.
  - `mental_health`: Scores for depression, stress, anxiety, and confidence.
  - `voice_analysis`: Stress and anxiety scores from voice analysis (if audio is present).

#### Example Request (cURL)
```bash
curl -X POST "https://<your-username>-EvolviSense.hf.space/analyze-video/" \
  -F "file=@/path/to/video.webm" \
  -F "frame_skip=10" \
  -F "max_frames=50"
```

#### Example Response
```json
{
  "emotions": {
    "happy": [79.62, 99.47, 99.75, 47.32, 99.99, 99.96, 99.96, 99.99, 99.96, 99.97],
    "fear": [76.90, 44.78, 44.86, 74.60, 53.23, 74.49]
  },
  "mental_health": {
    "depression": 0.0,
    "stress": 19.5,
    "anxiety": 23.7,
    "confidence": 0
  },
  "voice_analysis": {
    "stress": 65.0,
    "anxiety": 79.0
  }
}
```

## Latest Updates
**Last Updated: May 6, 2025**

- **Enhanced Audio Extraction**: Implemented a fallback mechanism using `subprocess.run` with FFmpeg to extract audio when `MoviePy` fails, ensuring robust voice analysis for stress and anxiety detection.
- **Improved Frame Processing**: Optimized the handling of video frames to accurately map emotions with consistent logging for debugging.
- **Error Handling**: Strengthened error handling for video uploads, audio extraction, and emotion detection, with detailed logs to troubleshoot issues like file size limits or missing audio streams.

## Requirements
The API depends on the following Python packages (listed in `requirements.txt`):
- `fastapi`
- `uvicorn`
- `python-multipart`
- `opencv-python`
- `deepface`
- `librosa`
- `moviepy`
- `soundfile`
- `numpy`

**Note**: `DeepFace` may require additional dependencies like `tensorflow` or `tf-keras`, which are installed automatically. FFmpeg is required for audio extraction.

## Setup and Deployment
This Space is deployed using Docker on Hugging Face Spaces. The required files are:
- `app.py`: The FastAPI application code for video analysis.
- `requirements.txt`: List of Python dependencies.
- `Dockerfile`: Docker configuration for building the environment.
- `README.md`: This documentation.

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/EvolviSense
   cd EvolviSense
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install FFmpeg**:
   - Ubuntu/Debian: `sudo apt update && sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.
4. **Run the API**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```
5. **Test the API**:
   - Use the Swagger UI at `http://localhost:7860/docs` to test the `/analyze-video/` endpoint.
   - Upload a video file via cURL or a client like Postman.

### Docker Deployment
Use the following `Dockerfile` to deploy on Hugging Face Spaces or locally:
```dockerfile
FROM python:3.12
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install -y libhdf5-dev libgl1 ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
COPY --chown=user ./requirements.txt requirements.txt 
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```
Build and run the Docker container:
```bash
docker build -t evolvisense .
docker run -p 7860:7860 evolvisense
```

## Notes
- **Model Download**: `DeepFace` automatically downloads pre-trained models (e.g., VGG-Face) on the first request, which may cause a delay.
- **File Size Limit**: Videos must be under 10MB to avoid HTTP 413 errors.
- **Audio Extraction**: Ensure FFmpeg is installed and the video contains an audio stream for voice analysis.
- **Face Detection**: Videos must contain visible faces for accurate emotion detection; otherwise, neutral emotions are returned.
- **Troubleshooting**:
  - Check logs for errors related to FFmpeg or `DeepFace` model downloads.
  - Verify video format compatibility (WebM, MP4 recommended).
  - Monitor resource usage on Hugging Face Spaces to avoid timeouts.



