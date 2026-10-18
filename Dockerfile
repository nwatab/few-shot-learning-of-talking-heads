# TensorFlow 1.x has no arm64 wheels; on Apple Silicon build with --platform linux/amd64.
FROM python:3.6-slim

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
