"""
Unit tests for Auto-STAIR Retrieval Module.
Tests LeafTrie validation (0% hallucination guarantee), BM25 baseline retriever,
and LangGraph Multi-Hop Agent with mocked Groq responses.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.leaf_store import LeafSection, LeafStore
from src.ingestion.toc_parser import ToCNode, ToCTree
from src.retrieval.trie_validator import LeafTrie, ValidationResult
from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.multihop_agent import MultiHopSTAIRAgent


def test_leaf_trie_exact_and_fuzzy_snapping():
    """Verify that LeafTrie resolves exact, prefix, and fuzzy inputs strictly to LN_D."""
    leaf_titles = [
        "1.1 Memory Allocation",
        "1.2 Virtual Memory",
        "2.1 Page Replacement",
        "2.2 Thrashing",
    ]
    trie = LeafTrie(leaf_titles)

    # 1. Exact match
    res1 = trie.validate_and_snap("1.1 Memory Allocation")
    assert res1.is_valid is True
    assert res1.matched_leaf_title == "1.1 Memory Allocation"
    assert res1.match_strategy == "exact"

    # 2. Markdown / prompt prefix stripping
    res2 = trie.validate_and_snap("### Section: leaf 1.2 Virtual Memory")
    assert res2.is_valid is True
    assert res2.matched_leaf_title == "1.2 Virtual Memory"

    # 3. Partial prefix match
    res3 = trie.validate_and_snap("Page Replacement")
    assert res3.is_valid is True
    assert res3.matched_leaf_title == "2.1 Page Replacement"

    # 4. Fuzzy typo recovery
    res4 = trie.validate_and_snap("2.2 Trashin")
    assert res4.is_valid is True
    assert res4.matched_leaf_title == "2.2 Thrashing"
    assert res4.match_strategy in ("prefix", "fuzzy_recovery")

    # 5. Wild hallucination snap: guaranteed to snap to a valid leaf in LN_D (0% hallucination)
    res5 = trie.validate_and_snap("Quantum Computing Superconductors")
    assert res5.is_valid is True
    assert res5.matched_leaf_title in leaf_titles


def test_bm25_baseline_retrieval():
    """Verify BM25 baseline retrieval scores and ranking."""
    sections = [
        LeafSection(
            doc_id="os_book",
            book_title="Operating Systems",
            leaf_id="os_book#1",
            tree_path="Memory > Paging",
            leaf_title="Paging Mechanics",
            section_text="Paging is a memory management scheme that eliminates the need for contiguous allocation.",
            page_start=1,
            page_end=3,
        ),
        LeafSection(
            doc_id="os_book",
            book_title="Operating Systems",
            leaf_id="os_book#2",
            tree_path="Processes > Scheduling",
            leaf_title="CPU Scheduling",
            section_text="CPU scheduling deals with round robin, first-come first-serve, and priority queues.",
            page_start=4,
            page_end=6,
        ),
    ]

    bm25 = BM25Retriever(sections)
    results = bm25.retrieve("How does paging memory allocation work?", top_k=2)

    assert len(results) == 2
    top_sec, score = results[0]
    assert top_sec.leaf_title == "Paging Mechanics"
    assert score > 0.0


def test_multihop_agent_with_mocked_groq():
    """Verify LangGraph multi-hop agent execution flow with mocked Groq responses."""
    nodes = [
        ToCNode(node_id=0, level=1, title="Chapter 1: Storage", page=1, children_ids=[1, 2], tree_path="Chapter 1: Storage"),
        ToCNode(node_id=1, level=2, title="Section 1.1: File Systems", page=2, parent_id=0, tree_path="Chapter 1: Storage > Section 1.1: File Systems"),
        ToCNode(node_id=2, level=2, title="Section 1.2: Inodes", page=5, parent_id=0, tree_path="Chapter 1: Storage > Section 1.2: Inodes"),
    ]
    tree = ToCTree(doc_id="test_fs", book_title="Storage Systems", nodes=nodes)

    store = LeafStore(":memory:")
    sec1 = LeafSection(
        doc_id="test_fs",
        book_title="Storage Systems",
        leaf_id="test_fs#1",
        tree_path="Chapter 1: Storage > Section 1.1: File Systems",
        leaf_title="Section 1.1: File Systems",
        section_text="File systems organize files on block devices using directories and metadata.",
        page_start=2,
        page_end=4,
    )
    sec2 = LeafSection(
        doc_id="test_fs",
        book_title="Storage Systems",
        leaf_id="test_fs#2",
        tree_path="Chapter 1: Storage > Section 1.2: Inodes",
        leaf_title="Section 1.2: Inodes",
        section_text="An inode stores file attributes and direct/indirect block pointers.",
        page_start=5,
        page_end=8,
    )
    store.add_sections([sec1, sec2])

    agent = MultiHopSTAIRAgent(groq_api_key="gsk_mock_valid_12345")

    # Mock Groq responses:
    # 1. Router picks "Section 1.2: Inodes"
    # 2. Grader answers "YES" (sufficient)
    # 3. Synthesizer answers grounded answer with citation
    responses = [
        "Section 1.2: Inodes",
        "YES",
        "Based on [Chapter 1: Storage > Section 1.2: Inodes], an inode contains direct and indirect pointers.",
    ]
    
    with patch.object(agent, "_call_llm", side_effect=responses):
        res = agent.run(
            query="What pointers does an inode contain?",
            doc_id="test_fs",
            book_title="Storage Systems",
            toc_tree=tree,
            leaf_store=store,
        )

        assert "Inodes" in res["visited_leaf_titles"][0]
        assert len(res["retrieved_sections"]) == 1
        assert res["is_sufficient"] is True
        assert "[Chapter 1: Storage > Section 1.2: Inodes]" in res["citations"][0]
        assert "direct and indirect pointers" in res["final_answer"]
