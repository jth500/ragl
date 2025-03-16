import argparse
import os

from ..ingest.csv import CSVIngestor
from ..retrieve.vector_store import ChromaVectorStore


def ingest_csv_files(
    directory: str,
    text_column: str,
    chroma_db_path: str,
    metadata_fields=None,
    chunk_size=500,
):
    """
    Ingests all CSV files in a given directory into ChromaDB.

    Args:
        directory (str): Path to the directory containing CSV files.
        chroma_db_path (str): Path to the ChromaDB file storage.
        metadata_fields (list, optional): List of column names to include as metadata. Defaults to None.
        chunk_size (int, optional): Number of characters per chunk. Defaults to 500.
    """
    store = ChromaVectorStore(db_path=chroma_db_path)

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")

            # Read CSV and process
            ingestor = CSVIngestor(
                file_path=file_path,
                text_column=text_column,
                chunk_size=chunk_size,
                metadata_fields=metadata_fields,
            )
            ingestor.read_file()
            chunks = ingestor.chunk_text()
            store.add_documents(chunks)
            print(f"{filename} added to ChromaDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest CSV files into ChromaDB.")
    parser.add_argument(
        "directory", type=str, help="Path to the directory containing CSV files."
    )
    parser.add_argument("text_column", type=str, help="Name of the text column.")
    parser.add_argument(
        "chroma_db_path", type=str, help="Path to the ChromaDB file storage."
    )
    parser.add_argument(
        "--metadata_fields",
        nargs="*",
        default=None,
        help="List of column names to include as metadata.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500, help="Chunk length for text splitting."
    )

    args = parser.parse_args()
    ingest_csv_files(
        args.directory,
        args.text_column,
        args.chroma_db_path,
        args.metadata_fields,
        args.chunk_size,
    )
