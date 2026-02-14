FROM python:3.12-slim

LABEL maintainer="Preetham S"
LABEL description="Donizo Truth Engine V0.1 — The Bloomberg for Construction"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default entrypoint
ENTRYPOINT ["python", "-m", "donizo_engine"]
