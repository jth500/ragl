# ragl

## Overview
ragl is a local-first RAG pipeline using ChromaDB for storage and Ollama for LLM inference. It supports document ingestion, chunking, vector storage, and querying via a FastAPI interface.

## Features
- **Document ingestion** with metadata
- **Vector search** using ChromaDB
- **Flexible embeddings** (Ollama, SentenceTransformers, or custom models)
- **FastAPI interface** for querying
- **Dockerized** for ease

## Setup & Usage
### Clone the Repository
```sh
git clone https://github.com/jth500/ragl.git
cd ragl
```

### Running with Docker
Ensure **Docker** is installed, then build and run the container:
```sh
docker-compose up --build -d
```

### Ingesting data (CSV for now)

1. **Copy CSV files into the container:**
   ```sh
   docker cp my_data.csv ragl:/app/data/my_data.csv
   ```

2. **Enter the Docker container shell:**
   ```sh
   docker exec -it ragl /bin/sh
   ```

3. **Run the ingestion script:**
   ```sh
   python -m ragl.scripts.ingest_csv /app/data text_col_name /app/chroma_db 
   ```


### Querying the API
```sh
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What do we think of Tottenham?", "top_k": 3, "model": "llama3.2"}'
```


## Roadmap
- Support for additional document formats (PDF, JSON, etc.)
- Cleaner data ingestion


