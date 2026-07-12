FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies needed for OpenCV/Albumentations
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/animal_recognition && touch /app/animal_recognition/__init__.py
COPY *.py /app/

ENV PYTHONPATH=/app:$PYTHONPATH

CMD ["python", "app.py"]
