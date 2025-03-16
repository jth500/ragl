FROM python:3.10-slim
WORKDIR /app
RUN pip install "poetry==1.7.1"

# Copy and install dependencies
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

# Copy source code and everything 
COPY ragl/ /app/ragl/
RUN mkdir -p /app/chroma_db
VOLUME /app/chroma_db

# Expose API port and run it all
EXPOSE 8000
CMD ["uvicorn", "ragl.main:app", "--host", "0.0.0.0", "--port", "8000"]
