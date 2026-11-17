"""
Leaf Store Module for Auto-STAIR.
Uses Python's built-in sqlite3 database to store intact leaf sections without arbitrary chunking.
"""

from __future__ import annotations
import sqlite3
from typing import List, Optional
from pydantic import BaseModel, Field


class LeafSection(BaseModel):
    """Represents an intact terminal leaf section extracted from a structured document."""
    doc_id: str = Field(description="Unique document identifier")
    book_title: str = Field(description="Title of the document or book")
    leaf_id: str = Field(description="Unique identifier for the leaf node (e.g., doc_id#leaf_index)")
    tree_path: str = Field(description="Hierarchical breadcrumb path (e.g., 'Chapter 1 > Section 1.2 > Title')")
    leaf_title: str = Field(description="Terminal section heading title")
    section_text: str = Field(description="Intact, un-chunked section content")
    page_start: int = Field(default=1, description="1-indexed starting page number")
    page_end: int = Field(default=1, description="1-indexed ending page number")


class LeafStore:
    """SQLite-backed persistent store for intact terminal leaf sections."""

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the SQLite leaf store.
        
        Args:
            db_path: Path to SQLite database file or ':memory:' for transient storage.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return self._conn

    def _init_db(self) -> None:
        """Create the leaf_sections schema if it does not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS leaf_sections (
                    doc_id TEXT NOT NULL,
                    book_title TEXT NOT NULL,
                    leaf_id TEXT PRIMARY KEY,
                    tree_path TEXT NOT NULL,
                    leaf_title TEXT NOT NULL,
                    section_text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_leaf_doc_id ON leaf_sections(doc_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_leaf_title ON leaf_sections(leaf_title)"
            )
            conn.commit()

    def add_section(self, section: LeafSection) -> None:
        """Insert or replace an intact leaf section."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO leaf_sections
                (doc_id, book_title, leaf_id, tree_path, leaf_title, section_text, page_start, page_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    section.doc_id,
                    section.book_title,
                    section.leaf_id,
                    section.tree_path,
                    section.leaf_title,
                    section.section_text,
                    section.page_start,
                    section.page_end,
                ),
            )
            conn.commit()

    def add_sections(self, sections: List[LeafSection]) -> None:
        """Batch insert or replace intact leaf sections."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO leaf_sections
                (doc_id, book_title, leaf_id, tree_path, leaf_title, section_text, page_start, page_end)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.doc_id,
                        s.book_title,
                        s.leaf_id,
                        s.tree_path,
                        s.leaf_title,
                        s.section_text,
                        s.page_start,
                        s.page_end,
                    )
                    for s in sections
                ],
            )
            conn.commit()

    def get_section(self, leaf_id: str) -> Optional[LeafSection]:
        """Fetch a single leaf section by its unique leaf_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM leaf_sections WHERE leaf_id = ?", (leaf_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return LeafSection(**dict(row))

    def get_section_by_title(self, leaf_title: str, doc_id: Optional[str] = None) -> Optional[LeafSection]:
        """Fetch a leaf section by exact leaf title matching."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute(
                    "SELECT * FROM leaf_sections WHERE doc_id = ? AND leaf_title = ? LIMIT 1",
                    (doc_id, leaf_title),
                )
            else:
                cursor.execute(
                    "SELECT * FROM leaf_sections WHERE leaf_title = ? LIMIT 1",
                    (leaf_title,),
                )
            row = cursor.fetchone()
            if row is None:
                return None
            return LeafSection(**dict(row))

    def get_all_sections(self, doc_id: Optional[str] = None) -> List[LeafSection]:
        """Fetch all leaf sections, optionally filtered by doc_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute(
                    "SELECT * FROM leaf_sections WHERE doc_id = ? ORDER BY page_start ASC",
                    (doc_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM leaf_sections ORDER BY page_start ASC"
                )
            rows = cursor.fetchall()
            return [LeafSection(**dict(row)) for row in rows]

    def get_all_leaf_titles(self, doc_id: Optional[str] = None) -> List[str]:
        """Get the ordered list of all terminal leaf titles (LN_D)."""
        sections = self.get_all_sections(doc_id)
        return [s.leaf_title for s in sections]

    def clear(self, doc_id: Optional[str] = None) -> None:
        """Clear sections from the store."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if doc_id:
                cursor.execute("DELETE FROM leaf_sections WHERE doc_id = ?", (doc_id,))
            else:
                cursor.execute("DELETE FROM leaf_sections")
            conn.commit()
