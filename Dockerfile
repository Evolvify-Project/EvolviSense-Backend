FROM python:3.12

RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies for tensorflow

RUN apt-get update && apt-get install -y libhdf5-dev libgl1 ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

COPY --chown=user ./requirements.txt requirements.txt 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
