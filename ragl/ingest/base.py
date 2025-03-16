import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional


class BaseIngestor(ABC):
    """
    Abstract base class for processing a single file.
    Handles metadata extraction and text chunking.
    """

    def __init__(
        self,
        file_path: str,
        chunk_size: int = 500,
        document_metadata: Optional[Dict] = {},
    ):
        """
        Initializes the base ingestor.

        Args:
            file_path (str): Path to the file to be processed.
            chunk_size (int, optional): Number of tokens per chunk. Defaults to 500.
            document_metadata (Optional[Dict], optional): metadata for the document. Defaults to {}.
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.document_metadata = document_metadata

    def read_file(self, path):
        # return the file object
        pass

    def process(self) -> List[Dict]:
        """
        Extracts, chunks, and processes the file.

        Returns:
            List[Dict]: List of dictionaries containing chunked text and metadata.
        """
        extracted_data = self.extract_text()
        return self.chunk_text(extracted_data)
