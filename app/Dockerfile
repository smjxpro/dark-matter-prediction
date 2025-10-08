# --- Base image ---
FROM python:3.12-slim

# --- System dependencies for scientific stack ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Set working directory ---
WORKDIR /app

# --- Copy dependencies ---
COPY requirements.txt .

# --- Upgrade pip and install dependencies ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.nvidia.com

# --- Copy app ---
COPY . .

# --- Run the app ---
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
