# DocuGraph AI - Multi-stage Dockerfile
# Supports both development and production builds

# ============================================================
# Stage 1: Base image with Python and system dependencies
# ============================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    # For Playwright/Crawl4AI
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

# Increase UV timeout for large packages
ENV UV_HTTP_TIMEOUT=300

# ============================================================
# Stage 2: Development image
# ============================================================
FROM base as development

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install CPU-only PyTorch first (avoid heavy CUDA deps in Docker)
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies including dev extras
RUN uv pip install --system -e ".[dev]"

# Install Playwright browsers for Crawl4AI
RUN playwright install chromium --with-deps

# Create data directory
RUN mkdir -p /data/docugraph

# Set data directory environment variable
ENV DOCUGRAPH_STORAGE__DATA_DIR=/data/docugraph

# Expose ports
EXPOSE 8000 3000

# Default command
CMD ["docugraph", "--help"]

# ============================================================
# Stage 3: Production image (optimized)
# ============================================================
FROM base as production

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install CPU-only PyTorch first (avoid heavy CUDA deps in Docker)
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Install only production dependencies
RUN uv pip install --system -e .

# Install Playwright browsers for Crawl4AI
RUN playwright install chromium --with-deps

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash docugraph \
    && mkdir -p /data/docugraph \
    && chown -R docugraph:docugraph /app /data/docugraph

USER docugraph

# Set data directory environment variable
ENV DOCUGRAPH_STORAGE__DATA_DIR=/data/docugraph

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the API server
CMD ["docugraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
