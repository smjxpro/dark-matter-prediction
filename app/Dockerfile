FROM python:3.12-alpine

# Install build tools for numpy/pandas/scikit-learn
RUN apk add --no-cache gcc g++ musl-dev libffi-dev cmake make

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
