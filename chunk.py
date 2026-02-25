"""
AutoArchitect — Code Chunking

Splits source files into overlapping 50-line windows.
Reads ingested_files.json, writes chunks.json.
"""

import json
import os
import sys

# Configuration

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
MIN_CHUNK_LINES = 3
INCLUDE_FILE_HEADER = True


def chunk_file(document: dict) -> list:
    """Split a single file into overlapping chunks with metadata."""

    file_path = document["file_path"]
    content = document["content"]
    language = document["language"]
    lines = content.split("\n")
    total_lines = len(lines)

    # Small files get returned as-is
    if total_lines <= CHUNK_SIZE:
        chunk_content = content
        if INCLUDE_FILE_HEADER:
            header = f"File: {file_path} | Language: {language}"
            chunk_content = header + "\n\n" + content

        return [{
            "content": chunk_content,
            "file_path": file_path,
            "language": language,
            "start_line": 1,
            "end_line": total_lines,
            "chunk_index": 0,
            "total_chunks_in_file": 1
        }]
    
    # Slide a window across the file
    chunks = []
    step_size = CHUNK_SIZE - CHUNK_OVERLAP
    chunk_index = 0

    for start in range(0, total_lines, step_size):
        end = min(start + CHUNK_SIZE, total_lines)

        chunk_lines = lines[start : end]
        
        # Skip tiny leftover fragments
        if len(chunk_lines) < MIN_CHUNK_LINES:
            continue

        chunk_content = "\n".join(chunk_lines)

        # Header helps the embedding model know where this code is from
        if INCLUDE_FILE_HEADER:
            header = f"File: {file_path} | Language: {language} | Lines: {start + 1}-{end}"
            chunk_content = header + "\n\n" + chunk_content

        chunks.append({
            "content": chunk_content,
            "file_path": file_path,
            "language": language,
            "start_line": start + 1,
            "end_line": end,
            "chunk_index": chunk_index,
        })

        chunk_index += 1

        if end == total_lines:
            break

    for chunk in chunks:
        chunk["total_chunks_in_file"] = len(chunks)

    return chunks

def chunk_all_files(documents: list) -> list:
    """Chunk every file and assign globally unique IDs."""

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        file_chunks = chunk_file(doc)

        # Unique ID across the whole codebase
        for chunk in file_chunks:
            chunk["chunk_id"] = chunk_id
            chunk_id += 1

        all_chunks.extend(file_chunks)

    return all_chunks

if __name__ == "__main__":
    with open("ingested_files.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunks = chunk_all_files(documents)

    with open("chunks.json", "w", encoding="utf-8") as out:
        json.dump(chunks, out, indent=2, ensure_ascii=False)

    print(f"Done! {len(chunks)} chunks saved.")