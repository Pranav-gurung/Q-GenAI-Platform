FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml
RUN pip install .

COPY alembic.ini ./
COPY scripts ./scripts
COPY frontend ./frontend
COPY data/samples ./data/samples
COPY knowledge ./knowledge
COPY docker/entrypoint.sh /entrypoint.sh

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data/raw/images /app/data/rag && chown -R app /app/data \
    && chmod +x /entrypoint.sh
USER app

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "backend.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
