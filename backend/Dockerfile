FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV SAM2_BUILD_CUDA=0
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH="/app/external/sam2:/app:${PYTHONPATH}"

RUN apt-get update && apt-get install -y \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY external/sam2 ./external/sam2
COPY app ./app

RUN mkdir -p uploads results checkpoints

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
