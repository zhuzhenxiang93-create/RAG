FROM python:3.11-slim

WORKDIR /app
COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt
RUN useradd --create-home --uid 10001 docmind \
    && mkdir -p /var/lib/docmind \
    && chown -R docmind:docmind /var/lib/docmind
COPY --chown=docmind:docmind . .

ENV DOCMIND_APP_MODE=lite
ENV DOCMIND_DATA_DIR=/var/lib/docmind
ENV DOCMIND_LOG_FORMAT=json
EXPOSE 8000
USER docmind
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
