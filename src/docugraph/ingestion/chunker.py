"""Document chunking strategies."""

import re
from dataclasses import dataclass
from enum import StrEnum

from docugraph.core.models import Chunk, Document


class ChunkingStrategy(StrEnum):
    """Available chunking strategies."""

    FIXED = "fixed"  # Fixed size with overlap
    SEMANTIC = "semantic"  # By markdown headings/sections
    SENTENCE = "sentence"  # By sentences
    PARAGRAPH = "paragraph"  # By paragraphs


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""

    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 2000  # Maximum chunk size


class Chunker:
    """Document chunker with multiple strategies."""

    def __init__(self, config: ChunkerConfig | None = None):
        """Initialize the chunker.

        Args:
            config: Chunker configuration.
        """
        self._config = config or ChunkerConfig()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Chunk a document into smaller pieces.

        Args:
            document: The document to chunk.

        Returns:
            List of chunks.
        """
        strategy = self._config.strategy

        if strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(document)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentences(document)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraphs(document)
        else:
            return self._chunk_fixed(document)

    def _chunk_fixed(self, document: Document) -> list[Chunk]:
        """Chunk document with fixed size windows."""
        content = document.content
        chunks: list[Chunk] = []

        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap

        start = 0
        while start < len(content):
            end = start + chunk_size

            # Try to break at sentence/paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start + chunk_size // 2, end + 100)
                if para_break > start:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = content.rfind(". ", start + chunk_size // 2, end + 50)
                    if sent_break > start:
                        end = sent_break + 1

            chunk_content = content[start:end].strip()

            if len(chunk_content) >= self._config.min_chunk_size:
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=chunk_content,
                        start_char=start,
                        end_char=end,
                        metadata={
                            "source_url": document.source_url,
                            "source_path": document.source_path,
                            "title": document.title,
                            "chunk_index": len(chunks),
                        },
                    )
                )

            start = end - overlap
            if start >= len(content):
                break

        return chunks

    def _chunk_semantic(self, document: Document) -> list[Chunk]:
        """Chunk document by markdown sections."""
        content = document.content
        chunks: list[Chunk] = []

        # Split by headings
        heading_pattern = r"^(#{1,6})\s+(.+?)$"
        sections: list[tuple[int, int, str, str]] = []  # (start, level, title, content)

        lines = content.split("\n")
        current_section_start = 0
        current_title = document.title or "Introduction"
        current_level = 0
        section_lines: list[str] = []

        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line)
            if match:
                # Save previous section
                if section_lines:
                    section_content = "\n".join(section_lines).strip()
                    if section_content:
                        sections.append(
                            (
                                current_section_start,
                                current_level,
                                current_title,
                                section_content,
                            )
                        )

                # Start new section
                current_level = len(match.group(1))
                current_title = match.group(2).strip()
                current_section_start = sum(len(line) + 1 for line in lines[:i])
                section_lines = [line]
            else:
                section_lines.append(line)

        # Add last section
        if section_lines:
            section_content = "\n".join(section_lines).strip()
            if section_content:
                sections.append(
                    (
                        current_section_start,
                        current_level,
                        current_title,
                        section_content,
                    )
                )

        # Convert sections to chunks, splitting large ones
        for start, level, title, section_content in sections:
            if len(section_content) <= self._config.max_chunk_size:
                if len(section_content) >= self._config.min_chunk_size:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            content=section_content,
                            start_char=start,
                            end_char=start + len(section_content),
                            metadata={
                                "source_url": document.source_url,
                                "source_path": document.source_path,
                                "title": document.title,
                                "section_title": title,
                                "heading_level": level,
                                "chunk_index": len(chunks),
                            },
                        )
                    )
            else:
                # Split large sections with fixed chunking
                sub_chunks = self._split_large_section(
                    document, section_content, start, title, level, len(chunks)
                )
                chunks.extend(sub_chunks)

        # If no chunks created (no headings found), fall back to fixed chunking
        if not chunks:
            return self._chunk_fixed(document)

        return chunks

    def _split_large_section(
        self,
        document: Document,
        content: str,
        base_start: int,
        title: str,
        level: int,
        chunk_index_start: int,
    ) -> list[Chunk]:
        """Split a large section into smaller chunks."""
        chunks: list[Chunk] = []

        # Try to split by paragraphs first
        paragraphs = content.split("\n\n")
        current_chunk = ""
        current_start = base_start
        chunk_idx = chunk_index_start

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self._config.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk
                if len(current_chunk) >= self._config.min_chunk_size:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            content=current_chunk,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            metadata={
                                "source_url": document.source_url,
                                "source_path": document.source_path,
                                "title": document.title,
                                "section_title": title,
                                "heading_level": level,
                                "chunk_index": chunk_idx,
                            },
                        )
                    )
                    chunk_idx += 1

                current_start = base_start + content.find(para)
                current_chunk = para

        # Add last chunk
        if current_chunk and len(current_chunk) >= self._config.min_chunk_size:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=current_chunk,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        "source_url": document.source_url,
                        "source_path": document.source_path,
                        "title": document.title,
                        "section_title": title,
                        "heading_level": level,
                        "chunk_index": chunk_idx,
                    },
                )
            )

        return chunks

    def _chunk_sentences(self, document: Document) -> list[Chunk]:
        """Chunk document by sentences, grouping to target size."""
        content = document.content
        chunks: list[Chunk] = []

        # Simple sentence splitting
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, content)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self._config.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if len(current_chunk) >= self._config.min_chunk_size:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            content=current_chunk,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            metadata={
                                "source_url": document.source_url,
                                "source_path": document.source_path,
                                "title": document.title,
                                "chunk_index": len(chunks),
                            },
                        )
                    )

                current_start = content.find(sentence, current_start)
                current_chunk = sentence

        # Add last chunk
        if current_chunk and len(current_chunk) >= self._config.min_chunk_size:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=current_chunk,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        "source_url": document.source_url,
                        "source_path": document.source_path,
                        "title": document.title,
                        "chunk_index": len(chunks),
                    },
                )
            )

        return chunks

    def _chunk_paragraphs(self, document: Document) -> list[Chunk]:
        """Chunk document by paragraphs, grouping to target size."""
        content = document.content
        chunks: list[Chunk] = []

        paragraphs = content.split("\n\n")

        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self._config.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if len(current_chunk) >= self._config.min_chunk_size:
                    chunks.append(
                        Chunk(
                            document_id=document.id,
                            content=current_chunk,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            metadata={
                                "source_url": document.source_url,
                                "source_path": document.source_path,
                                "title": document.title,
                                "chunk_index": len(chunks),
                            },
                        )
                    )

                current_start = content.find(para, current_start)
                current_chunk = para

        # Add last chunk
        if current_chunk and len(current_chunk) >= self._config.min_chunk_size:
            chunks.append(
                Chunk(
                    document_id=document.id,
                    content=current_chunk,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        "source_url": document.source_url,
                        "source_path": document.source_path,
                        "title": document.title,
                        "chunk_index": len(chunks),
                    },
                )
            )

        return chunks


def get_chunker(config: ChunkerConfig | None = None) -> Chunker:
    """Get a chunker instance."""
    return Chunker(config)
