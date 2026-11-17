"""
Hierarchical PDF Outline Parser and Dynamic ToC Inducer for Auto-STAIR.
Implements Section 4.1 (Formal Tree Definitions) and Section 7/8 (Dynamic Structure Induction).
"""

from __future__ import annotations
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from src.ingestion.leaf_store import LeafSection


class ToCNode(BaseModel):
    """Represents a node in the Table of Contents tree (ToC_D)."""
    node_id: int = Field(description="Sequential 0-indexed ID")
    level: int = Field(description="Hierarchy depth (1 = Chapter, 2 = Section, etc.)")
    title: str = Field(description="Heading or section title")
    page: int = Field(description="1-indexed starting page number in document")
    parent_id: Optional[int] = Field(default=None, description="Parent node ID, or None if root")
    children_ids: List[int] = Field(default_factory=list, description="List of direct child node IDs")
    tree_path: str = Field(default="", description="Full breadcrumb path (e.g., 'Chapter 1 > Section 1.2 > Topic')")

    @property
    def is_leaf(self) -> bool:
        """A node is a leaf in LN_D if it has zero child nodes."""
        return len(self.children_ids) == 0


class ToCTree(BaseModel):
    """Complete hierarchical Table of Contents tree (ToC_D) of a document."""
    doc_id: str
    book_title: str
    total_pages: int = Field(default=1, description="Total pages in the source document")
    nodes: List[ToCNode] = Field(default_factory=list)

    @property
    def leaf_nodes(self) -> List[ToCNode]:
        """Set of terminal leaf nodes LN_D with zero children."""
        return [node for node in self.nodes if node.is_leaf]

    @property
    def leaf_titles(self) -> List[str]:
        """List of valid terminal leaf titles."""
        return [node.title for node in self.leaf_nodes]

    def get_node_by_title(self, title: str) -> Optional[ToCNode]:
        """Find a node by exact title."""
        title_clean = title.strip().lower()
        for node in self.nodes:
            if node.title.strip().lower() == title_clean:
                return node
        return None

    def format_for_prompt(self, leaf_prefix: bool = True) -> str:
        """
        Format the tree for the STAIR Table 2 prompt.
        Leaves are explicitly marked (e.g. with [leaf] or indented) to constrain model selection.
        """
        lines = []
        for node in self.nodes:
            indent = "  " * (node.level - 1)
            marker = " [leaf]" if (node.is_leaf and leaf_prefix) else ""
            lines.append(f"{indent}- {node.title}{marker}")
        return "\n".join(lines)


class DynamicToCInducer:
    """
    Fallback ToC inducer for PDFs lacking native bookmark outlines.
    Extracts font size/weight candidates and optionally queries Groq llama-3.1-8b-instant
    to synthesize a valid hierarchical ToC.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

    def extract_heading_candidates(self, doc: fitz.Document, max_pages: int = 400) -> List[Dict[str, Any]]:
        """
        Extract candidate headings based on font size and boldness.
        Scans up to max_pages to support full enterprise textbooks and multi-hundred-page specifications.
        """
        candidates: List[Dict[str, Any]] = []
        font_sizes: List[float] = []

        total_pages = min(len(doc), max_pages)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            blocks = page.get_text("dict", flags=fitz.TEXT_DEHYPHENATE).get("blocks", [])

            for b in blocks:
                if b.get("type") == 0:  # text block
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            size = round(span.get("size", 10.0), 1)
                            flags = span.get("flags", 0)
                            is_bold = bool(flags & 2 or flags & 16 or "bold" in span.get("font", "").lower())
                            if len(text) > 2 and len(text) < 120:
                                font_sizes.append(size)
                                candidates.append({
                                    "text": text,
                                    "size": size,
                                    "is_bold": is_bold,
                                    "page": page_num,
                                })

        if not font_sizes:
            return []

        # Baseline body text: use 25th percentile (or min font size) to avoid heading skew
        font_sizes.sort()
        baseline_size = font_sizes[len(font_sizes) // 4]
        max_size = font_sizes[-1]

        # Filter candidates significantly larger than baseline or bold
        filtered = []
        seen = set()
        for c in candidates:
            is_heading_candidate = (
                c["is_bold"]
                or c["size"] > baseline_size * 1.15
                or (max_size > baseline_size and c["size"] >= max_size * 0.85)
            )
            if is_heading_candidate and c["text"].lower() not in seen:
                # Exclude purely numeric or page header/footer noise
                if not re.match(r"^\d+$", c["text"]):
                    seen.add(c["text"].lower())
                    filtered.append(c)

        return filtered

    def induce_toc(self, doc: fitz.Document, book_title: str) -> List[Tuple[int, str, int]]:
        """
        Induce a hierarchical ToC. Returns list of (level, title, page).
        Falls back to rule-based font size clustering if Groq is unavailable or offline.
        """
        candidates = self.extract_heading_candidates(doc)
        if not candidates:
            # Fallback: create single root section
            return [(1, f"{book_title} - Full Document", 1)]

        # Try Groq API induction if key is provided and valid
        if self.api_key and not self.api_key.startswith("mock_"):
            try:
                from groq import Groq
                client = Groq(api_key=self.api_key)
                
                # Sample top 40 candidates
                sample_candidates = candidates[:40]
                prompt = (
                    f"You are a document structure induction specialist. "
                    f"Given candidate headings extracted from the document '{book_title}', "
                    f"synthesize a clean, 2-level Table of Contents hierarchy (level 1 for Chapters/Major Parts, "
                    f"level 2 for Sections/Subsections).\n\n"
                    f"Candidates:\n{json.dumps(sample_candidates, indent=2)}\n\n"
                    f"Output strictly valid JSON with key 'toc' containing a list of objects with "
                    f"'level' (integer 1 or 2), 'title' (string), and 'page' (integer)."
                )

                models_to_try = ["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
                response = None
                for m in models_to_try:
                    try:
                        response = client.chat.completions.create(
                            model=m,
                            messages=[
                                {"role": "system", "content": "You output only valid JSON."},
                                {"role": "user", "content": prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                        )
                        if response:
                            break
                    except Exception as merr:
                        if "model_not_found" in str(merr).lower() or "does not exist" in str(merr).lower():
                            continue
                        break

                if response:
                    data = json.loads(response.choices[0].message.content)
                    toc_items = data.get("toc", [])
                    if toc_items:
                        return [(int(item["level"]), str(item["title"]), int(item["page"])) for item in toc_items]
            except Exception:
                # If Groq fails or is offline, seamlessly use deterministic rule-based fallback
                pass

        # Deterministic font-based clustering fallback (100% offline & fast)
        sizes = sorted(list(set(c["size"] for c in candidates)), reverse=True)
        max_size = sizes[0] if sizes else 14.0
        
        toc_result: List[Tuple[int, str, int]] = []
        for c in candidates[:30]:
            level = 1 if c["size"] >= max_size * 0.9 else 2
            toc_result.append((level, c["text"], c["page"]))

        return toc_result


class PDFToCParser:
    """
    Parses native outlines or induces dynamic ToCs from PDFs,
    builds the hierarchical tree ToC_D, identifies leaf nodes LN_D,
    and extracts intact text sections without arbitrary token slicing.
    """

    def __init__(self, inducer: Optional[DynamicToCInducer] = None):
        self.inducer = inducer or DynamicToCInducer()

    def parse_pdf(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        book_title: Optional[str] = None,
    ) -> Tuple[ToCTree, List[LeafSection]]:
        """
        Parse a PDF, build the ToCTree, and extract intact leaf sections.

        Args:
            pdf_path: Path to the target PDF file.
            doc_id: Unique identifier for the document (defaults to filename).
            book_title: Human-readable book/document title.

        Returns:
            Tuple of (ToCTree, List[LeafSection]).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        actual_doc_id = doc_id or os.path.splitext(os.path.basename(pdf_path))[0]
        actual_book_title = book_title or actual_doc_id.replace("_", " ").title()

        # 1. Extract raw ToC: [(level, title, page), ...]
        raw_toc = doc.get_toc()  # returns [[lvl, title, page], ...]
        if not raw_toc:
            # Empty native outline: activate DynamicToCInducer
            raw_toc = self.inducer.induce_toc(doc, actual_book_title)

        # 2. Build ToCNode objects and parent-child tree
        toc_tree = self._build_tree(actual_doc_id, actual_book_title, raw_toc, total_pages=len(doc))

        # 3. Extract intact leaf section texts
        sections = self._extract_leaf_sections(doc, toc_tree)
        doc.close()

        return toc_tree, sections

    def _build_tree(
        self, doc_id: str, book_title: str, raw_toc: List[Tuple[int, str, int] | List[Any]], total_pages: int = 1
    ) -> ToCTree:
        """
        Constructs the formal ToC_D tree from raw (level, title, page) records.
        Calculates parent_id, children_ids, and hierarchical breadcrumb paths.
        """
        nodes: List[ToCNode] = []
        parent_stack: List[Tuple[int, int]] = []  # (level, node_id)

        for idx, item in enumerate(raw_toc):
            level = int(item[0])
            title = str(item[1]).strip()
            page = max(1, int(item[2]))

            # Find parent: top of stack with level < current level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            parent_id = parent_stack[-1][1] if parent_stack else None

            node = ToCNode(
                node_id=idx,
                level=level,
                title=title,
                page=page,
                parent_id=parent_id,
            )

            # Register child with parent
            if parent_id is not None:
                nodes[parent_id].children_ids.append(idx)

            nodes.append(node)
            parent_stack.append((level, idx))

        # Build full breadcrumb tree paths
        for node in nodes:
            path_parts = [node.title]
            curr = node
            while curr.parent_id is not None:
                curr = nodes[curr.parent_id]
                path_parts.append(curr.title)
            node.tree_path = " > ".join(reversed(path_parts))

        return ToCTree(doc_id=doc_id, book_title=book_title, total_pages=total_pages, nodes=nodes)

    def _extract_leaf_sections(
        self, doc: fitz.Document, toc_tree: ToCTree
    ) -> List[LeafSection]:
        """
        Extracts intact section texts for all leaf nodes LN_D.
        Each section spans from its leaf start page until the next ToC node's page (or doc end).
        Never splits or chunks into arbitrary 500-token slices!
        """
        total_pages = len(doc)
        leaf_sections: List[LeafSection] = []
        all_nodes = toc_tree.nodes

        for node in toc_tree.leaf_nodes:
            start_page = node.page
            
            # Find next node in document sequence to bound the page range
            next_page = total_pages + 1
            for later_node in all_nodes[node.node_id + 1:]:
                if later_node.page > start_page:
                    next_page = later_node.page
                    break

            end_page = max(start_page, min(total_pages, next_page - 1 if next_page <= total_pages else total_pages))

            # Extract un-chunked page text from start_page to end_page (1-indexed to 0-indexed)
            text_parts = []
            for p in range(start_page - 1, end_page):
                if 0 <= p < total_pages:
                    text_parts.append(doc[p].get_text())

            full_text = "\n".join(text_parts).strip()
            if not full_text:
                full_text = f"[{node.title} - Section text]"

            leaf_sections.append(
                LeafSection(
                    doc_id=toc_tree.doc_id,
                    book_title=toc_tree.book_title,
                    leaf_id=f"{toc_tree.doc_id}#leaf_{node.node_id}",
                    tree_path=node.tree_path,
                    leaf_title=node.title,
                    section_text=full_text,
                    page_start=start_page,
                    page_end=end_page,
                )
            )

        return leaf_sections
