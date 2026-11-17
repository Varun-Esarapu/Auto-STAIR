"""
Unit tests for Auto-STAIR Ingestion Module (toc_parser.py and leaf_store.py).
Tests native ToC parsing, intact text extraction, leaf store persistence,
and mocked dynamic ToC induction.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
import fitz  # PyMuPDF
import pytest

from src.ingestion.leaf_store import LeafSection, LeafStore
from src.ingestion.toc_parser import ToCNode, ToCTree, PDFToCParser, DynamicToCInducer


def test_leaf_store_crud():
    """Test SQLite leaf store operations in memory."""
    store = LeafStore(":memory:")
    
    sec1 = LeafSection(
        doc_id="doc1",
        book_title="Test Book",
        leaf_id="doc1#leaf_1",
        tree_path="Chapter 1 > Section 1.1",
        leaf_title="Section 1.1",
        section_text="This is the full text of section 1.1 without any chunking.",
        page_start=1,
        page_end=3,
    )
    sec2 = LeafSection(
        doc_id="doc1",
        book_title="Test Book",
        leaf_id="doc1#leaf_2",
        tree_path="Chapter 1 > Section 1.2",
        leaf_title="Section 1.2",
        section_text="This is section 1.2 text intact.",
        page_start=4,
        page_end=5,
    )
    
    store.add_sections([sec1, sec2])
    
    fetched = store.get_section("doc1#leaf_1")
    assert fetched is not None
    assert fetched.leaf_title == "Section 1.1"
    assert "without any chunking" in fetched.section_text
    
    titles = store.get_all_leaf_titles("doc1")
    assert titles == ["Section 1.1", "Section 1.2"]
    
    by_title = store.get_section_by_title("Section 1.2")
    assert by_title is not None
    assert by_title.page_start == 4


def test_toctree_hierarchy_and_leaves():
    """Verify formal parent-child tree construction and leaf node set LN_D."""
    # Hierarchy:
    # 1. Chapter 1 (not leaf, has children)
    #    2. Section 1.1 (leaf)
    #    2. Section 1.2 (not leaf, has child)
    #       3. Subsection 1.2.1 (leaf)
    # 1. Chapter 2 (leaf)
    raw_toc = [
        [1, "Chapter 1", 1],
        [2, "Section 1.1", 2],
        [2, "Section 1.2", 4],
        [3, "Subsection 1.2.1", 5],
        [1, "Chapter 2", 8],
    ]
    
    parser = PDFToCParser()
    tree = parser._build_tree("doc_test", "Test Tome", raw_toc)
    
    assert len(tree.nodes) == 5
    leaf_titles = tree.leaf_titles
    assert leaf_titles == ["Section 1.1", "Subsection 1.2.1", "Chapter 2"]
    
    # Check tree paths
    sec1 = tree.get_node_by_title("Section 1.1")
    assert sec1 is not None
    assert sec1.tree_path == "Chapter 1 > Section 1.1"
    
    sub121 = tree.get_node_by_title("Subsection 1.2.1")
    assert sub121 is not None
    assert sub121.tree_path == "Chapter 1 > Section 1.2 > Subsection 1.2.1"
    
    prompt_str = tree.format_for_prompt(leaf_prefix=True)
    assert "- Section 1.1 [leaf]" in prompt_str
    assert "- Subsection 1.2.1 [leaf]" in prompt_str
    assert "[leaf]" not in prompt_str.split("\n")[0]  # Chapter 1 is not a leaf


def test_pdf_parsing_with_synthetic_pdf():
    """Create a temporary PDF with PyMuPDF and test end-to-end native parsing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create a synthetic 4-page PDF with an embedded outline
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Chapter 1 Introduction content on page 1")
        
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Section 1.1 Details and explanation on page 2")
        
        p3 = doc.new_page()
        p3.insert_text((50, 72), "Chapter 2 Advanced Methods content on page 3")
        
        p4 = doc.new_page()
        p4.insert_text((50, 72), "Section 2.1 Experiments and results on page 4")
        
        # Native outline:
        # Chapter 1 (page 1)
        #   Section 1.1 (page 2)
        # Chapter 2 (page 3)
        #   Section 2.1 (page 4)
        doc.set_toc([
            [1, "Chapter 1", 1],
            [2, "Section 1.1", 2],
            [1, "Chapter 2", 3],
            [2, "Section 2.1", 4],
        ])
        doc.save(tmp_path)
        doc.close()
        
        parser = PDFToCParser()
        tree, sections = parser.parse_pdf(tmp_path, doc_id="synth_book", book_title="Synthetic Book")
        
        assert len(sections) == 2
        assert sections[0].leaf_title == "Section 1.1"
        assert sections[1].leaf_title == "Section 2.1"
        assert "Section 1.1 Details" in sections[0].section_text
        assert "Section 2.1 Experiments" in sections[1].section_text
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dynamic_toc_inducer_mocked_groq():
    """Verify fallback ToC induction using mocked Groq response."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Introduction to Systems", fontsize=20)
        p1.insert_text((50, 120), "Body text here.")
        
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Memory Management", fontsize=18)
        p2.insert_text((50, 120), "More body text.")
        
        doc.save(tmp_path)
        doc.close()
        
        # Mock Groq API
        mock_groq_resp = MagicMock()
        mock_groq_resp.choices = [
            MagicMock(message=MagicMock(content='{"toc": [{"level": 1, "title": "Introduction to Systems", "page": 1}, {"level": 2, "title": "Memory Management", "page": 2}]}'))
        ]
        
        with patch("groq.Groq") as mock_groq_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_groq_resp
            mock_groq_cls.return_value = mock_client
            
            inducer = DynamicToCInducer(groq_api_key="gsk_mock_valid_key_12345")
            doc = fitz.open(tmp_path)
            raw_toc = inducer.induce_toc(doc, "OS Principles")
            doc.close()
            
            assert len(raw_toc) == 2
            assert raw_toc[0] == (1, "Introduction to Systems", 1)
            assert raw_toc[1] == (2, "Memory Management", 2)
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_universal_doc_parser_docx_and_markdown():
    """Verify that docx and markdown files are parsed into ToC trees and leaf sections."""
    parser = PDFToCParser()

    # 1. Test Markdown
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as tmp_md:
        tmp_md.write("# Overview\nThis is the overview paragraph.\n\n## Architecture\nDetails of the architecture.\n\n## Experiments\nResults shown here.\n")
        tmp_md_path = tmp_md.name

    try:
        tree, sections = parser.parse_document(tmp_md_path, doc_id="test_md", book_title="Markdown Specification")
        assert len(tree.nodes) == 3
        assert len(sections) == 2  # Architecture and Experiments are leaves
        assert sections[0].leaf_title == "Architecture"
        assert "Details of the architecture" in sections[0].section_text
    finally:
        if os.path.exists(tmp_md_path):
            os.remove(tmp_md_path)

    # 2. Test DOCX
    import docx
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
        tmp_docx_path = tmp_docx.name

    try:
        doc = docx.Document()
        doc.add_heading("Part I: System Overview", level=1)
        doc.add_paragraph("Introductory text about the system.")
        doc.add_heading("Module 1.1: Core Engine", level=2)
        doc.add_paragraph("Core engine implementation details.")
        doc.save(tmp_docx_path)

        tree, sections = parser.parse_document(tmp_docx_path, doc_id="test_docx", book_title="Docx Specification")
        assert len(tree.nodes) == 2
        assert len(sections) == 1  # Module 1.1 is leaf
        assert sections[0].leaf_title == "Module 1.1: Core Engine"
        assert "Core engine implementation" in sections[0].section_text
    finally:
        if os.path.exists(tmp_docx_path):
            os.remove(tmp_docx_path)


def test_dynamic_toc_inducer_deterministic_offline_fallback():
    """Verify deterministic font-clustering fallback when no Groq API key is present."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Major Chapter Header", fontsize=24)
        p1.insert_text((50, 120), "Regular standard text line.", fontsize=11)
        
        p2 = doc.new_page()
        p2.insert_text((50, 72), "Sub Section Header", fontsize=16)
        p2.insert_text((50, 120), "Another regular line.", fontsize=11)
        
        doc.save(tmp_path)
        doc.close()
        
        inducer = DynamicToCInducer(groq_api_key="")  # No key -> deterministic fallback
        doc = fitz.open(tmp_path)
        raw_toc = inducer.induce_toc(doc, "Fallback Tome")
        doc.close()
        
        assert len(raw_toc) >= 1
        titles = [t[1] for t in raw_toc]
        assert any("Major Chapter Header" in t for t in titles)
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
