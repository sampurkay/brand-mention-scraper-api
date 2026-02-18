# Playwright base image includes Chromium + OS deps
# [Unverified] Tag availability depends on Playwright release registry; if this tag fails,
# use the closest available tag (e.g., v1.57.0-jammy) or a newer version.
FROM mcr.microsoft.com/playwright/python:v1.57.0-jammy

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Railway provides PORT; default fallback
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Start FastAPI (package layout: app/main.py)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
