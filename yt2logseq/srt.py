from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
import pysrt  # noqa:F401


class StampedSRTLoader(BaseLoader):
    """Load `.srt` (subtitle) files."""

    def __init__(self, file_path: str, data: Optional[str]=None):
        """Initialize with a file path."""
        self.file_path = file_path
        self.data = data


    def load(self, max_chunk_size: int = 512) -> List[Document]:
        """Load subtitles by chunks.
        Each chunk contains at most `chunk_size` tokens.
        Chunks always end at the end of a subtitle.

        Parameters
        ----------
        chunk_size
            The desired number of tokens in each chunks.
        """
        if self.data:
            srt = pysrt.from_string(self.data)
        else:
            srt = pysrt.open(self.file_path)
        chunk_text = ""
        chunk_size = 0
        chunk_start = srt[0].start.to_time()
        chunks  = []
        for line in srt:
            line_size = len(line.text.split(" "))
            if (chunk_size + line_size) > max_chunk_size:
                metadata = {
                    "source": self.file_path,
                    'start': chunk_start,
                    'end': line.end.to_time()
                }
                chunks += [Document(page_content=chunk_text, metadata=metadata)]
                chunk_text = ""
                chunk_size = 0
                chunk_start = line.start.to_time()
            chunk_text += str(line)
            chunk_size += line_size
        try:
            metadata = {
                'source': self.file_path,
                'start': chunk_start,
                'end': line.end.to_time()
            }
            chunks += [Document(page_content=chunk_text, metadata=metadata)]
            return chunks
        except UnboundLocalError:
            return []
