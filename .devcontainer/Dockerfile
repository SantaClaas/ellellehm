FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Install ffmpeg to merge audio files using pydub
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pipx install uv