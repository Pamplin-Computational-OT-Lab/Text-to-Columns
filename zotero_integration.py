"""
zotero_integration.py

This script queries a local Zotero SQLite DB and retrieves metadata
for items in a specified collection, including:
- itemID
- title
- authors
- year
- pdf_path (first local attachment, if any)
"""

import sqlite3
import os
import re

def get_all_collections(db_path):
    """
    Returns a sorted list of all collection names from the Zotero database.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Zotero database not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT collectionName FROM collections")
    rows = c.fetchall()
    conn.close()

    collection_names = [r[0] for r in rows if r[0] is not None]
    return sorted(set(collection_names))

def get_collection_items_metadata(db_path, collection_name, require_attachment=False):
    """
    Retrieves item metadata (title, authors, year) plus the folder name
    for local PDFs in the specified Zotero collection.

    Args:
        db_path (str): Path to Zotero sqlite database.
        collection_name (str): Exact name of the collection.
        require_attachment (bool): 
            If True, only return items with a local PDF attachment.

    Returns:
        list of dict: For each item:
            "itemID" (int),
            "title" (str),
            "authors" (str),
            "year" (str),
            "key" (str): Zotero folder name containing the PDF.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Zotero database not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1) Find the collectionID
    c.execute("SELECT collectionID FROM collections WHERE collectionName=?", (collection_name,))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    collection_id = row[0]

    # 2) From collectionItems, get all itemIDs in that collection
    c.execute("SELECT itemID FROM collectionItems WHERE collectionID=?", (collection_id,))
    item_ids = [r[0] for r in c.fetchall()]
    if not item_ids:
        conn.close()
        return []

    all_metadata = []

    for item_id in item_ids:
        # If require_attachment is True, skip items with no local PDF
        if require_attachment:
            c.execute("""
                SELECT key
                FROM items
                WHERE itemID=?
            """, (item_id,))
            row_key = c.fetchone()
            if not row_key:
                continue

        # 3) Fetch key (folder name for the PDF)
        c.execute("""
            SELECT key
            FROM items
            WHERE itemID=?
        """, (item_id,))
        row_key = c.fetchone()
        key = row_key[0] if row_key else ""

        # 4) Fetch title (fieldID=110)
        c.execute("""
            SELECT itemDataValues.value
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            WHERE itemData.itemID = ?
              AND itemData.fieldID = 110
        """, (item_id,))
        row_title = c.fetchone()
        title = row_title[0] if row_title else ""

        # 5) Fetch authors (group_concat of last names)
        c.execute("""
            SELECT group_concat(creators.lastName, '; ')
            FROM itemCreators
            JOIN creators ON itemCreators.creatorID = creators.creatorID
            WHERE itemCreators.itemID = ?
        """, (item_id,))
        row_authors = c.fetchone()
        authors = row_authors[0] if row_authors else ""

        # 6) Fetch year (fieldID=115)
        c.execute("""
            SELECT itemDataValues.value
            FROM itemData
            JOIN itemDataValues ON itemData.valueID = itemDataValues.valueID
            WHERE itemData.itemID = ?
              AND itemData.fieldID = 115
        """, (item_id,))
        row_date = c.fetchone()
        date_val = row_date[0] if row_date else ""
        year = ""
        if date_val:
            m = re.match(r"(\d{4})", date_val)
            if m:
                year = m.group(1)

        # Build the metadata dictionary
        meta_dict = {
            "itemID": item_id,
            "title": title,
            "authors": authors,
            "year": year,
            "key": key  # Folder name for the PDF
        }
        all_metadata.append(meta_dict)

    conn.close()
    return all_metadata

