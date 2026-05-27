"""
Paper Parser Module for WuYa System

This module implements the PaperParser class for parsing academic papers
from various input formats (text, PDF) into structured ParsedPaper objects.

Features:
- PDF text extraction (with PyPDF2 support)
- Structured field extraction (title, abstract, keywords, sections, references)
- Caching of parsed results
- Mock mode for testing

Author: WuYa Team
"""

import re
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from .base import ParsedPaper

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Management
# =============================================================================

@dataclass
class ParseCacheEntry:
    """Cache entry for parsed papers."""
    paper_id: str
    parsed_paper: ParsedPaper
    timestamp: datetime = field(default_factory=datetime.now)
    source_hash: str = ""


class ParseCache:
    """Simple in-memory cache for parsed papers."""

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, ParseCacheEntry] = {}
        self._max_size = max_size

    def get(self, source_hash: str) -> Optional[ParsedPaper]:
        """Get cached parsed paper by source hash."""
        entry = self._cache.get(source_hash)
        if entry:
            logger.debug(f"Cache hit for hash: {source_hash[:8]}...")
            return entry.parsed_paper
        return None

    def set(self, source_hash: str, parsed_paper: ParsedPaper):
        """Cache a parsed paper."""
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest entry")

        self._cache[source_hash] = ParseCacheEntry(
            paper_id=parsed_paper.paper_id,
            parsed_paper=parsed_paper,
            source_hash=source_hash
        )
        logger.debug(f"Cached paper with hash: {source_hash[:8]}...")

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Parse cache cleared")


# =============================================================================
# PDF Extractor Interface
# =============================================================================

class PDFExtractor(ABC):
    """Abstract base class for PDF text extractors."""

    @abstractmethod
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from PDF file."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the extractor is available."""
        pass


class PyPDF2Extractor(PDFExtractor):
    """PDF text extractor using PyPDF2."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if PyPDF2 is available."""
        if self._available is None:
            try:
                import PyPDF2
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from PDF using PyPDF2."""
        if not self.is_available():
            raise ImportError("PyPDF2 is not installed. Install with: pip install PyPDF2")

        import PyPDF2

        text_parts = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text() or "")
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

        return "\n".join(text_parts)


class MockPDFExtractor(PDFExtractor):
    """Mock PDF extractor for testing."""

    def is_available(self) -> bool:
        return True

    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """Return mock text for testing."""
        return f"""
Mock PDF Content from {pdf_path}

Title: Mock Paper Title
Authors: John Doe, Jane Smith

Abstract:
This is a mock abstract for testing purposes. It describes the paper's
contributions and findings in a concise manner.

Keywords: machine learning, deep learning, neural networks

1. Introduction
This is the introduction section. It provides background and motivation.

2. Methodology
This section describes the methods used in the research.

3. Results
The results section presents the findings.

4. Discussion
This section discusses the implications.

5. Conclusion
The conclusion summarizes the paper.

References:
[1] Author et al. (2020). Important paper.
[2] Researcher (2019). Another paper.
"""


# =============================================================================
# Paper Parser Implementation
# =============================================================================

class PaperParser:
    """
    Parser for academic papers.

    Extracts structured information from raw paper text or PDF files,
    producing ParsedPaper objects for downstream evaluation.

    Features:
    - PDF text extraction (with fallback to mock)
    - Structured field extraction (title, abstract, keywords, etc.)
    - Result caching
    - Extensible extractor interface

    Example::

        # Parse from text
        parser = PaperParser()
        paper = parser.parse_paper(raw_text, source_type="text")

        # Parse from PDF
        paper = parser.parse_paper("/path/to/paper.pdf", source_type="pdf")

        # With caching
        parser = PaperParser(enable_cache=True)
        paper1 = parser.parse_paper(raw_text)  # Parses
        paper2 = parser.parse_paper(raw_text)  # Returns cached result

    Example (with mock for testing)::

        parser = PaperParser(use_mock_pdf=True)
        paper = parser.parse_paper("any_path.pdf", source_type="pdf")
    """

    def __init__(
        self,
        enable_cache: bool = True,
        use_mock_pdf: bool = False,
        pdf_extractor: Optional[PDFExtractor] = None,
    ):
        """
        Initialize PaperParser.

        Args:
            enable_cache: Whether to cache parse results.
            use_mock_pdf: Whether to use mock PDF extractor for testing.
            pdf_extractor: Custom PDF extractor (overrides use_mock_pdf).
        """
        self._cache = ParseCache() if enable_cache else None

        # Set up PDF extractor
        if pdf_extractor:
            self._pdf_extractor = pdf_extractor
        elif use_mock_pdf:
            self._pdf_extractor = MockPDFExtractor()
        else:
            self._pdf_extractor = PyPDF2Extractor()

        self._enable_cache = enable_cache

    def parse_paper(
        self,
        raw_input: Union[str, Path],
        source_type: Optional[str] = None,
        paper_id: Optional[str] = None,
        discipline: str = "general",
    ) -> ParsedPaper:
        """
        Parse paper from raw input.

        Args:
            raw_input: Raw text string or path to PDF file.
            source_type: "text", "pdf", or None (auto-detect).
            paper_id: Optional paper ID (generated if not provided).
            discipline: Paper discipline/category.

        Returns:
            ParsedPaper with extracted fields.
        """
        # Auto-detect source type
        if source_type is None:
            source_type = self._detect_source_type(raw_input)

        # Generate source hash for caching
        source_hash = self._compute_hash(raw_input)

        # Check cache
        if self._cache and self._enable_cache:
            cached = self._cache.get(source_hash)
            if cached:
                return cached

        # Extract raw text
        if source_type == "pdf":
            raw_text = self._extract_from_pdf(raw_input)
        else:
            raw_text = str(raw_input)

        # Parse structured fields
        parsed = self._parse_structured_fields(raw_text, paper_id, discipline)

        # Cache result
        if self._cache and self._enable_cache:
            self._cache.set(source_hash, parsed)

        return parsed

    def parse_batch(
        self,
        inputs: List[Union[str, Path]],
        source_type: Optional[str] = None,
        discipline: str = "general",
    ) -> List[ParsedPaper]:
        """
        Parse multiple papers in batch.

        Args:
            inputs: List of raw texts or PDF paths.
            source_type: Source type for all inputs (or None for auto-detect).
            discipline: Default discipline for all papers.

        Returns:
            List of ParsedPaper objects.
        """
        results = []
        for i, raw_input in enumerate(inputs):
            paper_id = f"batch_{i}_{uuid.uuid4().hex[:8]}"
            try:
                paper = self.parse_paper(raw_input, source_type, paper_id, discipline)
                results.append(paper)
            except Exception as e:
                logger.error(f"Failed to parse paper {i}: {e}")
                # Create minimal paper on error
                results.append(self._create_error_paper(str(e), paper_id))
        return results

    def clear_cache(self):
        """Clear the parse cache."""
        if self._cache:
            self._cache.clear()

    def _detect_source_type(self, raw_input: Union[str, Path]) -> str:
        """Detect whether input is text or PDF path."""
        if isinstance(raw_input, Path):
            return "pdf" if raw_input.suffix.lower() == ".pdf" else "text"

        input_str = str(raw_input)
        # Check if it's a file path
        if input_str.endswith(".pdf") or "/" in input_str or "\\" in input_str:
            path = Path(input_str)
            if path.exists() and path.suffix.lower() == ".pdf":
                return "pdf"

        return "text"

    def _extract_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from PDF file."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not self._pdf_extractor.is_available():
            logger.warning("PDF extractor not available, using mock")
            mock_extractor = MockPDFExtractor()
            return mock_extractor.extract_text(pdf_path)

        return self._pdf_extractor.extract_text(path)

    def _parse_structured_fields(
        self,
        raw_text: str,
        paper_id: Optional[str],
        discipline: str,
    ) -> ParsedPaper:
        """Parse structured fields from raw text."""
        # Generate paper ID if not provided
        if paper_id is None:
            paper_id = f"paper_{uuid.uuid4().hex[:12]}"

        # Extract fields
        title = self._extract_title(raw_text)
        abstract = self._extract_abstract(raw_text)
        keywords = self._extract_keywords(raw_text)
        authors = self._extract_authors(raw_text)
        sections = self._extract_sections(raw_text)
        references = self._extract_references(raw_text)

        return ParsedPaper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            content=raw_text,
            authors=authors,
            keywords=keywords,
            discipline=discipline,
            citations=[],  # Would need citation extraction
            references=references,
            figures=[],  # Would need figure extraction
            tables=[],  # Would need table extraction
        )

    def _extract_title(self, text: str) -> str:
        """Extract paper title from text."""
        # Try common patterns
        patterns = [
            r'(?:Title|TITLE)[:\s]+(.+?)(?:\n|$)',
            r'^\s*([^\n]{10,200})\s*(?:\n\s*(?:Author|Abstract|1\.|Introduction))',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fallback: first non-empty line that's reasonably long
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:10]:  # Check first 10 lines
            if 20 < len(line) < 200 and not line.startswith(('Abstract', 'Introduction')):
                return line

        return "Unknown Title"

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text."""
        # Try common patterns
        patterns = [
            r'(?:Abstract|ABSTRACT)[:\s]*(.+?)(?:\n\s*(?:Keywords|Key words|Introduction|1\.|\Z))',
            r'(?:Abstract|ABSTRACT)[:\s]*(.+?)(?=\n\s*\d+\.|\n\s*Introduction)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                return abstract[:2000]  # Limit length

        # Fallback: first substantial paragraph
        paragraphs = text.split('\n\n')
        for para in paragraphs[:3]:
            para = para.strip()
            if 100 < len(para) < 1500 and 'abstract' not in para.lower():
                return para

        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Try common patterns
        patterns = [
            r'(?:Keywords|Key words|KEYWORDS)[:\s]*(.+?)(?:\n|$)',
            r'(?:Keywords|Key words|KEYWORDS)[:\s]*(.+?)(?=\n\s*\d+\.|\n\s*Introduction)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1)
                # Split by common delimiters
                keywords = re.split(r'[,;]', keywords_text)
                return [k.strip() for k in keywords if k.strip() and len(k.strip()) > 2]

        # Fallback: extract from abstract
        abstract = self._extract_abstract(text)
        # Simple keyword extraction: words that appear capitalized
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', abstract)
        common_words = {'This', 'Paper', 'Study', 'Research', 'Method', 'Results', 'The', 'A', 'An'}
        keywords = [w for w in set(words) if w not in common_words]
        return keywords[:10]  # Limit to 10 keywords

    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors from text."""
        # Try common patterns
        patterns = [
            r'(?:Author|Authors|AUTHOR)[:\s]*(.+?)(?:\n|$)',
            r'(?:By|by)[:\s]*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                authors_text = match.group(1)
                # Split by common delimiters
                authors = re.split(r'[,;]', authors_text)
                return [a.strip() for a in authors if a.strip()]

        return []

    def _extract_sections(self, text: str) -> List[str]:
        """Extract section headings from text."""
        # Match common section patterns
        section_pattern = r'\n\s*(?:\d+\.|)([A-Z][A-Za-z\s]{2,50})[\n:]'
        matches = re.findall(section_pattern, text)

        # Filter out false positives
        common_sections = [
            'Introduction', 'Method', 'Methods', 'Methodology',
            'Results', 'Discussion', 'Conclusion', 'Conclusions',
            'Related Work', 'Background', 'Experiments',
            'Evaluation', 'Analysis', 'Future Work'
        ]

        sections = []
        for match in matches:
            match = match.strip()
            if any(common in match for common in common_sections) or len(match) < 50:
                sections.append(match)

        return sections[:20]  # Limit to 20 sections

    def _extract_references(self, text: str) -> List[str]:
        """Extract references from text."""
        # Try to find references section
        ref_section_pattern = r'(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[\s:]*(.+?)(?=\Z|\n\s*Appendix)'
        match = re.search(ref_section_pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            ref_text = match.group(1)
            # Split by reference numbers or newlines
            refs = re.split(r'\n\s*\[?\d+\]?[.\s]+|\n\s*\d+\.', ref_text)
            return [r.strip() for r in refs if len(r.strip()) > 20][:50]  # Limit to 50 refs

        return []

    def _compute_hash(self, raw_input: Union[str, Path]) -> str:
        """Compute hash of input for caching."""
        input_str = str(raw_input)
        return hashlib.sha256(input_str.encode()).hexdigest()[:32]

    def _create_error_paper(self, error_msg: str, paper_id: str) -> ParsedPaper:
        """Create a minimal paper object for error cases."""
        return ParsedPaper(
            paper_id=paper_id,
            title="Parse Error",
            abstract=f"Failed to parse paper: {error_msg}",
            content="",
            authors=[],
            keywords=[],
            discipline="unknown",
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_paper_parser(
    enable_cache: bool = True,
    use_mock_pdf: bool = False,
) -> PaperParser:
    """
    Factory function to create a PaperParser.

    Args:
        enable_cache: Whether to enable result caching.
        use_mock_pdf: Whether to use mock PDF extractor.

    Returns:
        Configured PaperParser instance.
    """
    return PaperParser(
        enable_cache=enable_cache,
        use_mock_pdf=use_mock_pdf,
    )
