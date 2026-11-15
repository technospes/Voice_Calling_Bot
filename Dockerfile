FROM python:3.11-slim

WORKDIR /app

# Install system dependencies, including ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from your requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Command to start your agent worker in production mode
CMD ["python", "-m", "app.agent.livekit_worker", "start"]