# FIX 1: Capitalize 'AS' to fix the linter warning
FROM python:3.11.9-slim AS builder
WORKDIR /usr/python/app/doc-analyzer
COPY requirements.txt ./

# FIX 2: Add 'build-essential' to install C/C++ compilers
# We also keep tesseract-ocr here in case a library needs to link against it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    tesseract-ocr \
    libtesseract-dev && \

    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# --- Final Stage ---
FROM python:3.11.9-slim
WORKDIR /usr/python/app/doc-analyzer

# Tesseract is needed here for runtime execution
RUN apt-get update && apt-get install -y tesseract-ocr && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m appuser && chown -R appuser:appuser /usr/python/app/doc-analyzer

# Copy installed packages from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN mkdir -p /usr/python/app/doc-analyzer/app/uploads && \
    chown -R appuser:appuser /usr/python/app/doc-analyzer/app && \
    chmod 755 /usr/python/app/doc-analyzer/app/uploads
USER appuser
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]