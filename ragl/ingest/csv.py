import os
import re
from typing import Dict, List, Optional

import pandas as pd
from tokenizers import Tokenizer

from .base import BaseIngestor


class CSVIngestor(BaseIngestor):
    """
    Processes a single CSV file into text chunks with row-level metadata.
    """

    def __init__(
        self,
        file_path: str,
        text_column: str,
        chunk_size: int = 500,
        metadata_fields: Optional[List[str]] = None,
        document_metadata: Optional[Dict] = None,
        tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initializes the CSV ingestor.

        Args:
            file_path (str): Path to the CSV file.
            text_column (str): Name of the column containing text data.
            chunk_size (int, optional): Number of tokens per chunk. Defaults to 500.
            metadata_fields (Optional[List[str]], optional): List of column names to include in metadata. Defaults to None.
            document_metadata (Optional[Dict], optional): Dictionary of user-defined metadata. Defaults to None.
            tokenizer (str): Hugging Face tokenizer model name. Defaults to ""sentence-transformers/all-MiniLM-L6-v2"".
        """
        super().__init__(file_path, chunk_size, document_metadata)
        self.text_column = text_column
        self.metadata_fields = metadata_fields or []
        self.tokenizer = tokenizer
        self.data = pd.DataFrame()

    # @property
    # def tokenizer(self):
    #     # lazy load the tokenizer
    #     if isinstance(self._tokenizer, str):
    #         self._tokenizer = Tokenizer.from_pretrained(self._tokenizer)
    #     return self._tokenizer

    # @tokenizer.setter
    # def tokenizer(self, tokenizer):
    #     self._tokenizer = tokenizer

    def read_file(self) -> pd.DataFrame:
        """Reads the CSV file and returns a DataFrame."""
        df = pd.read_csv(self.file_path)
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in CSV file.")

        if not set(self.metadata_fields).issubset(df.columns):
            raise ValueError("Not all metadata fields were found in CSV file.")

        self.data = df
        return self.data

    # TODO: Stop this splitting words
    def chunk_text(self, overlap: int = 50) -> List[Dict]:
        """
        Splits the text column into fixed-length chunks with overlap while preserving row-level and document-level metadata.

        Args:
            overlap (int): Number of characters to overlap between chunks. Defaults to 50.

        Returns:
            List[Dict]: List of dictionaries containing chunked text and metadata.
        """
        assert overlap < self.chunk_size, "Overlap must be less than chunk size."
        chunks = []

        for index, row in enumerate(self.data.itertuples(index=True)):
            row_metadata = {col: getattr(row, col) for col in self.metadata_fields}
            text = getattr(row, self.text_column)

            # Create base metadata shared across all chunks for this row
            base_metadata = {
                "source": os.path.basename(self.file_path),
                "file_path": self.file_path,
                "row_index": index,
                **self.document_metadata,
                **row_metadata,
            }

            # Split text into fixed-size chunks with overlap
            for i, start in enumerate(range(0, len(text), self.chunk_size - overlap)):
                chunk_text = text[start : start + self.chunk_size]
                chunk_metadata = {**base_metadata, "chunk_index": i}
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "text": [
                "This is the first example sentence. It is quite short.",
                "Here is another row with more text data. It contains additional information.",
                "The third row has an even longer passage that should be tokenized and split correctly.",
            ],
            "author": ["Alice", "Bob", "Charlie"],
            "category": ["News", "Blog", "Report"],
        }
    )

    # Save to a temporary CSV file for testing
    sample_csv_path = "sample_test.csv"
    data.to_csv(sample_csv_path, index=False)

    # Initialize the ingestor
    ingestor = CSVIngestor(
        file_path=sample_csv_path,
        text_column="text",
        chunk_size=60,  # Small chunk size for easy testing
        metadata_fields=["author", "category"],
        document_metadata={"project": "Test Run"},
    )

    # Read the file and run chunking
    ingestor.read_file()
    chunks = ingestor.chunk_text(2)

    # Display results
    print(f"Generated {len(chunks)} chunks.\n")
    for i, chunk in enumerate(chunks):  # Show first 5 chunks
        print(f"Chunk {i + 1}:")
        print("Text:", chunk["text"])
        print("Metadata:", chunk["metadata"])
        print("-" * 40)
