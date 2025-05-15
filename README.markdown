---
title: EvolviSense
emoji: 🏢
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: An API for emotion and mental health analysis from video uploads with voice analysis
---

# Emotion and Mental Health Analysis API

This Hugging Face Space hosts a **FastAPI** application that performs emotion and mental health analysis from uploaded video files. It leverages the `DeepFace` library to detect emotions (e.g., happy, sad, angry) from video frames and computes mental health scores (depression, stress, anxiety) based on both facial emotions and voice analysis. The API also provides frame-by-frame emotional trend data for visualization.

## Features
- **Emotion Detection from Video**: Analyzes emotions from pre-recorded MP4 video files using `DeepFace`.
- **Voice Analysis**: Extracts audio from videos and analyzes stress and anxiety using voice features like pitch and energy.
- **Mental Health Scores**: Computes scores for depression, stress, and anxiety based on emotion distributions and voice analysis.
- **Frame-by-Frame Trends**: Tracks stress, anxiety, and confidence over time for each analyzed frame.
- **Concurrency Handling**: Uses a threading lock to ensure only one video is processed at a time, preventing server overload.
- **API Integration**: Built with FastAPI, suitable for integration with modern frontend frameworks.

## How to Use

### Uploaded Video Analysis (HTTP POST)
- **Endpoint**: `POST /analyze-video/`
- **Description**: Upload an MP4 video file to analyze emotions, mental health scores, and voice-based stress/anxiety.
- **Parameters**:
  - `file` (required): Video file (MP4, max 100MB).
  - `frame_skip` (optional): Number of frames to skip between analyses (default: 10).
  - `max_frames` (optional): Maximum frames to analyze (default: 50).
- **Response**:
  - `emotions`: Dictionary of detected emotions and their confidences across frames.
  - `mental_health`: Scores for depression, stress, anxiety, and confidence.
  - `voice_analysis`: Stress and anxiety scores based on voice pitch and energy (if audio is present).
  - `frame_data`: Per-frame data with timestamps, stress, anxiety, and confidence.
  - `peak_stress`: The highest stress score detected in any frame.
  - `video_duration`: Duration of the video in seconds.

#### Example Request (cURL)
```bash
curl -X POST "https://moodydev-EvolviSense.hf.space/analyze-video/" \
  -F "file=@/path/to/video.mp4" \
  -F "frame_skip=10" \
  -F "max_frames=50"
```

#### Example Response
```json
{
  "emotions": {
    "happy": [85.2, 78.4, 92.1],
    "sad": [65.3, 70.2],
    "neutral": [50.0]
  },
  "mental_health": {
    "depression": 20.0,
    "stress": 15.0,
    "anxiety": 10.0,
    "confidence": 75.5
  },
  "voice_analysis": {
    "stress": 30.0,
    "anxiety": 25.0
  },
  "frame_data": [
    {
      "time": 0.33,
      "stress": 10.5,
      "anxiety": 8.2,
      "confidence": 85.2
    },
    {
      "time": 0.67,
      "stress": 12.0,
      "anxiety": 9.0,
      "confidence": 78.4
    }
  ],
  "peak_stress": 12.0,
  "video_duration": 5.0
}
```

## Mental Health Scores Calculation
- **Facial Analysis**:
  - Emotions are detected using `DeepFace` (e.g., happy, sad, angry).
  - Mental health scores are computed using weighted emotion contributions:
    - Depression: 50% sad, 20% fear, 10% anger, 10% disgust, 10% neutral.
    - Stress: 40% anger, 30% fear, 10% surprise, 10% disgust, 10% sad.
    - Anxiety: 50% fear, 20% surprise, 20% sad, 10% anger.
  - Scores are normalized to a 0-100 scale.
  - A "joy threshold" (50%) reduces negative emotion impact if happiness dominates.
- **Voice Analysis**:
  - Audio is extracted and analyzed using `librosa` for pitch and energy.
  - Stress: 50% pitch, 50% energy.
  - Anxiety: 70% pitch, 30% energy.
  - Final scores combine facial (70%) and voice (30%) analysis if audio is present.

## Requirements
The API depends on the following:
- **Python Packages** (approximate versions based on usage):
  - `fastapi==0.115.0`
  - `uvicorn==0.30.6`
  - `python-multipart==0.0.9`
  - `opencv-python==4.10.0.84`
  - `deepface==0.0.93`
  - `numpy==2.1.1`
  - `librosa==0.10.2`
  - `soundfile==0.12.1`
  - `moviepy==1.0.3`
  - `pydub==0.25.1`
  - `ffmpeg-python==0.2.0`
- **System Dependencies**:
  - **FFmpeg**: Required for audio extraction. Install it on your system:
    - Ubuntu: `sudo apt-get install ffmpeg`
    - macOS: `brew install ffmpeg`
    - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

## Setup and Deployment
This Space is deployed using Docker on Hugging Face Spaces. The required files are:
- `app.py`: The FastAPI application code.
- `requirements.txt`: List of Python dependencies.
- `Dockerfile`: Docker configuration for building the environment.
- `README.md`: This documentation.

To deploy locally or on another platform:
1. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/moodydev/EvolviSense
   ```
2. Install FFmpeg (see above).
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the API:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Test the API at `http://localhost:8000/docs`.

## Notes
- **Model Download**: The `DeepFace` library downloads pre-trained models (e.g., VGG-Face) on first use, which may delay the first request.
- **Resource Limits**: Hugging Face Spaces free tier has limited CPU/memory. Large videos (>100MB) may cause timeouts. Adjust `max_frames` or `frame_skip` to reduce load.
- **Face Detection**: Ensure the video contains visible faces for accurate emotion detection. If no faces are detected, the API defaults to "neutral" emotion.
- **Audio Extraction**: Videos without audio will skip voice analysis, relying solely on facial analysis.
- **Frame Processing**: The API processes up to 50 frames by default, skipping every 10 frames to optimize performance.

## Troubleshooting
- **FFmpeg Errors**: Ensure FFmpeg is installed and supports H.264/AAC codecs. Check logs for "FFmpeg is not installed" or "FFmpeg does not support H.264 or AAC codecs".
- **Audio Extraction Fails**: If logs show "Error extracting audio", ensure the video has an audio track and FFmpeg is correctly installed.
- **No Frames Processed**: If logs show "No frames processed", the video may be corrupted or unreadable by OpenCV. Test with a different MP4 file.
- **Server Overload**: If you get a 429 error ("Please wait, processing previous video"), wait until the current video finishes processing.
- **Model Download Fails**: Ensure internet access for `DeepFace` to download models. Check logs for network errors.
- **Timeout Issues**: For slow connections, the API may timeout during large video uploads. Test with smaller videos (<10MB) first.

## License
This project is licensed under the MIT License.