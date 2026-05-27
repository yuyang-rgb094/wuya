# =============================================================================
# WuYa Agents — Docker Image
# =============================================================================
# Build:  docker build -t wuya-agents .
# Run:    docker run --rm -it wuya-agents wuya --help
# =============================================================================

FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="WuYa Team"
LABEL description="WuYa (无涯) — Theory-driven academic paper evaluation system"
LABEL org.opencontainers.image.source="https://github.com/wuya-team/wuya"

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Stage 1: Builder ---
FROM base AS builder

COPY pyproject.toml ./
COPY wuya_agents/ ./wuya_agents/

# Install build dependencies and the package
RUN pip install --break-system-packages --no-cache-dir . && \
    pip install --break-system-packages --no-cache-dir .[all]

# --- Stage 2: Runtime ---
FROM base AS runtime

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY wuya_agents/ ./wuya_agents/
COPY pyproject.toml ./
COPY .env.example ./.env.example

# Create data directory for vector store persistence
RUN mkdir -p /app/data

# Copy entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["wuya", "--help"]
