FROM python:3.12-slim

WORKDIR /app/env

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.1" \
    "fastapi>=0.115.0" \
    "fastmcp>=2.0.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "gradio>=4.0.0"

# Copy environment code
COPY . /app/env

ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV KGQA_DATA_PATH="/app/env/data"
ENV KGQA_MAX_STEPS="30"
ENV ENABLE_WEB_INTERFACE="true"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
