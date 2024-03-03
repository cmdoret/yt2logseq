from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
import pysrt  # noqa:F401


class StampedSRTLoader(BaseLoader):
    """Load `.srt` (subtitle) files."""

    def __init__(self, file_path: str):
        """Initialize with a file path."""
        self.file_path = file_path

    def load(self, chunk_size: int = 512) -> List[Document]:
        """Load subtitles by chunks.
        Each chunk contains at most `chunk_size` tokens.
        Chunks always end at the end of a subtitle.

        Parameters
        ----------
        chunk_size
            The desired number of tokens in each chunks.
        """
        parsed_info = pysrt.open(self.file_path)
        current_chunk = ""
        chunk_start = parsed_info[0].start.to_time()
        chunks  = []
        for line in parsed_info:
            if len(current_chunk) + len(line.text) > chunk_size:
                metadata = {
                    "source": self.file_path,
                    'start': chunk_start,
                    'end': line.end.to_time()
                }
                chunks += [Document(page_content=current_chunk, metadata=metadata)]
                current_chunk = ""
                chunk_start = line.start.to_time()
            current_chunk += line.text + " "
        try:
            metadata = {
                'source': self.file_path,
                'start': chunk_start,
                'end': line.end.to_time()
            }
            chunks += [Document(page_content=current_chunk, metadata=metadata)]
            return chunks
        except UnboundLocalError:
            return []
