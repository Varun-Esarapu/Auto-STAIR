"""
Prefix-Tree (LeafTrie) Validator for Auto-STAIR.
Constrains zero-shot router outputs to valid terminal leaf nodes LN_D,
guaranteeing 0% structural hallucinations via exact prefix matching, fuzzy recovery, and snap.
"""

from __future__ import annotations
import re
import difflib
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of leaf title validation against LN_D."""
    is_valid: bool = Field(description="Whether the output resolved to a valid leaf node")
    original_prediction: str = Field(description="Raw text emitted by LLM")
    matched_leaf_title: Optional[str] = Field(default=None, description="Exact leaf title in LN_D")
    match_strategy: str = Field(default="none", description="exact | prefix | fuzzy_recovery | fallback")
    confidence: float = Field(default=0.0, description="Match score between 0.0 and 1.0")


class TrieNode:
    """A single node in the LeafTrie character prefix tree."""
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_terminal: bool = False
        self.full_leaf_title: Optional[str] = None


class LeafTrie:
    """
    Prefix tree index over valid terminal leaf nodes (LN_D).
    Eliminates structural hallucinations by ensuring predicted sections strictly map to LN_D.
    """

    def __init__(self, leaf_titles: Optional[List[str]] = None):
        self.root = TrieNode()
        self.valid_leaf_titles: Set[str] = set()
        self.normalized_map: Dict[str, str] = {}  # normalized -> original
        if leaf_titles:
            self.build_trie(leaf_titles)

    @staticmethod
    def normalize(text: str) -> str:
        """Strip markdown markers, quotes, prefixes (e.g. '### Section:', '[leaf]'), and extra spaces."""
        cleaned = text.strip()
        # Remove common prompt echoes
        cleaned = re.sub(r"^(###\s*Section:\s*|Section:\s*|leaf\s*:?\s*)", "", cleaned, flags=re.IGNORECASE)
        # Remove [leaf] annotation
        cleaned = re.sub(r"\[leaf\]", "", cleaned, flags=re.IGNORECASE)
        # Remove leading bullets or numbers
        cleaned = re.sub(r"^[\s*\-•\d\.\)]+\s*", "", cleaned)
        # Normalize whitespace and casing
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        return cleaned

    def insert(self, leaf_title: str) -> None:
        """Insert a leaf title into the trie."""
        title = leaf_title.strip()
        if not title:
            return
        self.valid_leaf_titles.add(title)
        norm = self.normalize(title)
        self.normalized_map[norm] = title

        curr = self.root
        for char in norm:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_terminal = True
        curr.full_leaf_title = title

    def build_trie(self, leaf_titles: List[str]) -> None:
        """Populate the trie with a list of terminal leaf titles."""
        self.root = TrieNode()
        self.valid_leaf_titles.clear()
        self.normalized_map.clear()
        for title in leaf_titles:
            self.insert(title)

    def validate_and_snap(self, prediction: str, min_fuzzy_ratio: float = 0.55) -> ValidationResult:
        """
        Validate an LLM prediction string against the trie.
        Applies a 4-tier resolution hierarchy:
        1. Exact match (normalized)
        2. Strict prefix match in Trie
        3. Fuzzy string matching / token overlap recovery
        4. Safe snap to top leaf (guaranteeing 0% non-existent section hallucination)
        """
        if not prediction or not self.valid_leaf_titles:
            fallback = next(iter(self.valid_leaf_titles)) if self.valid_leaf_titles else None
            return ValidationResult(
                is_valid=fallback is not None,
                original_prediction=prediction,
                matched_leaf_title=fallback,
                match_strategy="fallback",
                confidence=0.0,
            )

        norm_pred = self.normalize(prediction)

        # Tier 1: Exact normalized match
        if norm_pred in self.normalized_map:
            return ValidationResult(
                is_valid=True,
                original_prediction=prediction,
                matched_leaf_title=self.normalized_map[norm_pred],
                match_strategy="exact",
                confidence=1.0,
            )

        # Tier 2: Check if normalized prediction starts with or contains any valid leaf title
        for norm_leaf, orig_leaf in self.normalized_map.items():
            if norm_leaf == norm_pred:
                return ValidationResult(
                    is_valid=True,
                    original_prediction=prediction,
                    matched_leaf_title=orig_leaf,
                    match_strategy="exact",
                    confidence=1.0,
                )
            # LLM said "Section: 1.1 Intro to Math - Some note" -> leaf is "1.1 Intro to Math"
            if norm_leaf in norm_pred or norm_pred in norm_leaf:
                overlap_len = min(len(norm_leaf), len(norm_pred))
                return ValidationResult(
                    is_valid=True,
                    original_prediction=prediction,
                    matched_leaf_title=orig_leaf,
                    match_strategy="prefix",
                    confidence=overlap_len / max(len(norm_leaf), len(norm_pred)),
                )

        # Tier 2b: Prefix traversal in the Trie
        curr = self.root
        prefix_matched = True
        for char in norm_pred:
            if char in curr.children:
                curr = curr.children[char]
            else:
                prefix_matched = False
                break

        if prefix_matched:
            # Collect all terminals under curr node
            candidates = self._collect_terminals(curr)
            if candidates:
                # Pick the shortest or first match
                best_match = min(candidates, key=len)
                return ValidationResult(
                    is_valid=True,
                    original_prediction=prediction,
                    matched_leaf_title=best_match,
                    match_strategy="prefix",
                    confidence=0.9,
                )

        # Tier 3: Fuzzy Recovery via SequenceMatcher
        matches = difflib.get_close_matches(
            norm_pred, list(self.normalized_map.keys()), n=1, cutoff=min_fuzzy_ratio
        )
        if matches:
            best_leaf = self.normalized_map[matches[0]]
            score = difflib.SequenceMatcher(None, norm_pred, matches[0]).ratio()
            return ValidationResult(
                is_valid=True,
                original_prediction=prediction,
                matched_leaf_title=best_leaf,
                match_strategy="fuzzy_recovery",
                confidence=round(score, 3),
            )

        # Tier 4: Fallback to the highest-token-overlap leaf in LN_D (Snapping prevents hallucination)
        tokens_pred = set(norm_pred.split())
        best_overlap_title = None
        best_overlap_count = -1
        for norm_leaf, orig_leaf in self.normalized_map.items():
            overlap = len(tokens_pred.intersection(set(norm_leaf.split())))
            if overlap > best_overlap_count:
                best_overlap_count = overlap
                best_overlap_title = orig_leaf

        fallback_title = best_overlap_title or next(iter(self.valid_leaf_titles))
        return ValidationResult(
            is_valid=True,
            original_prediction=prediction,
            matched_leaf_title=fallback_title,
            match_strategy="fallback_snap",
            confidence=0.3,
        )

    def _collect_terminals(self, node: TrieNode) -> List[str]:
        """Collect all terminal leaf titles rooted at a node."""
        results = []
        if node.is_terminal and node.full_leaf_title:
            results.append(node.full_leaf_title)
        for child in node.children.values():
            results.extend(self._collect_terminals(child))
        return results
