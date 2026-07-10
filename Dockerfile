# Portable container — works on Render (Docker runtime) and Hugging Face Spaces.
# Binds to $PORT when the host sets one (Render), else 7860 (HF Spaces default).
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces expects the app on 7860; Render injects its own $PORT at runtime.
ENV PORT=7860
EXPOSE 7860

# Shell form so ${PORT} is expanded at start. Matches the Procfile/Blueprint:
# long timeout for screener requests, 1 worker (memory) + 8 threads (I/O).
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 300 --workers 1 --threads 8
