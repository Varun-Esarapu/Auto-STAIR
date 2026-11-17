"""
BM25 Lexical Baseline Retriever for Auto-STAIR.
Uses rank_bm25 (BM25Okapi) over the exact same leaf section texts for direct comparison
against STAIR's structure-aware routing.
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from src.ingestion.leaf_store import LeafSection


class BM25Retriever:
    """Lexical baseline retriever using BM25Okapi over intact leaf sections."""

    def __init__(self, sections: Optional[List[LeafSection]] = None):
        self.sections: List[LeafSection] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        if sections:
            self.index_sections(sections)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, fast regex tokenizer without heavy NLP overhead."""
        return re.findall(r"\w+", text.lower())

    def index_sections(self, sections: List[LeafSection]) -> None:
        """Build BM25 index over the provided leaf sections."""
        self.sections = list(sections)
        # Combine leaf title, tree path, and intact section text
        self.tokenized_corpus = [
            self.tokenize(f"{s.tree_path} {s.leaf_title} {s.section_text}")
            for s in self.sections
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[LeafSection, float]]:
        """
        Retrieve top_k leaf sections matching the query using BM25.

        Returns:
            List of (LeafSection, score) tuples ranked in descending order of relevance.
        """
        if not self.bm25 or not self.sections:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = list(self.bm25.get_scores(tokenized_query))
        # If standard Okapi zeroed out on small corpus N (where RSJ IDF log(1) == 0),
        # back up with term frequency overlap
        if max(scores, default=0.0) == 0.0:
            query_set = set(tokenized_query)
            scores = [
                float(sum(1 for token in doc_tokens if token in query_set))
                for doc_tokens in self.tokenized_corpus
            ]

        # Pair with sections and sort
        ranked = sorted(
            zip(self.sections, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(section, float(score)) for section, score in ranked[:top_k]]
