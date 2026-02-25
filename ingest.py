"""
AutoArchitect — Repository Ingestion

Clones a GitHub repo, filters for source/config/doc files,
reads their contents, and dumps everything to ingested_files.json.
"""

import os
import json
import shutil
import subprocess
import sys

# Configuration

# Which file types do we want to index?
ALLOWED_EXTENSIONS = {
    # Source code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb",
    # Configuration
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env.example",
    # Documentation
    ".md", ".rst",
    # Infrastructure
    ".dockerfile", ".tf",
}

# Which directories should we skip entirely?
SKIP_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".tox",
    ".eggs",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "vendor",
    "third_party",
    ".next",
    "coverage",
    ".coverage",
}

# Skip files larger than this (in kilobytes)
MAX_FILE_SIZE_KB = 100

SKIP_FILES = {
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "pnpm-lock.yaml",
    ".DS_Store",
    "thumbs.db",
}


def clone_repo(repo_url: str, target_dir: str) -> str:
    """Clone a repo. Skips if already cloned. Returns the path."""

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = os.path.join(target_dir, repo_name)

    if os.path.exists(clone_path):
        print(f"Repository already exists at {clone_path}, skipping clone.")
        return clone_path
    
    print(f"Cloning {repo_url} into {repo_name}...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, clone_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"ERROR cloning: {result.stderr}")
        sys.exit(1)

    print(f"Clone complete.")
    return clone_path


def should_skip_directory(dir_name: str) -> bool:
    """Skip junk directories like node_modules, .git, etc."""

    return dir_name in SKIP_DIRECTORIES or dir_name.startswith(".")
    
def should_include_file(file_path: str) -> bool:
    """Check filename, extension, and size. Returns True if we want this file."""

    file_name = os.path.basename(file_path)

    if file_name in SKIP_FILES:
        return False
    
    _, ext = os.path.splitext(file_name)
    ext = ext.lower()

    # Dockerfile, Makefile, etc. have no extension
    if file_name.lower() in ("dockerfile", "makefile", "rakefile"):
        return True
    
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > MAX_FILE_SIZE_KB:
            return False
    except OSError:
        return False
    
    return True

def walk_repository(repo_path: str) -> list:
    """Walk the repo tree and return all files that pass our filters."""

    collected_files = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for file_name in files:
            abs_path = os.path.join(root, file_name)
            
            if should_include_file(abs_path):
                rel_path = os.path.relpath(abs_path, repo_path)
                collected_files.append({
                    "path": rel_path,
                    "abs_path": abs_path,
                })

    return collected_files


def read_files(file_list: list) -> list:
    """Read each file and package it with metadata. Uses errors='replace' because
    real repos have mixed encodings that would otherwise crash the pipeline."""

    documents = []
    skipped = 0

    for file_info in file_list:
        try:
            with open(file_info["abs_path"], "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # nothing to search over
            if not content.strip():
                skipped += 1
                continue

            documents.append({
                "file_path": file_info["path"],
                "content": content,
                "num_lines": content.count("\n") + 1,
                "language": detect_language(file_info["path"]),
            })

        except Exception as e:
            print(f"WARNING: Could not read {file_info['path']}: {e}")

    if skipped > 0:
        print(f"Skipped {skipped} files (empty or unreadable)")

    return documents

def detect_language(file_path: str) -> str:
    """Map file extension to language name."""

    ext_to_language = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".json": "json",
        ".md": "markdown",
        ".rst": "restructuredtext",
        ".ini": "ini",
        ".cfg": "ini",
        ".tf": "terraform",
    }

    _, ext = os.path.splitext(file_path)
    return ext_to_language.get(ext.lower(), "unknown")


def ingest_repository(repo_url: str, output_path: str = "ingested_files.json") -> list:
    """Run the full ingestion pipeline: clone, filter, read, save."""

    print("*" * 60)
    print("Repository Ingestion")
    print("*" * 60)

    print("\nCloning repository...")
    repo_path = clone_repo(repo_url, target_dir="./repos")

    print("\nScanning files...")
    file_list = walk_repository(repo_path)
    print(f"Found {len(file_list)} files after filtering")

    print("\nReading file contents...")
    documents = read_files(file_list)

    total_lines = sum(doc["num_lines"] for doc in documents)
    languages = {}
    for doc in documents:
        lang = doc["language"]
        languages[lang] = languages.get(lang, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files indexed:  {len(documents)}")
    print(f"  Total lines:    {total_lines:,}")
    print(f"  Languages:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        print(f"{lang:20s} {count:5d} files")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print("Done!")

    return documents


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://github.com/encode/httpx"
        print(f"No URL provided. Using default: {url}")
        print(f"Usage: python ingest.py <github_url>\n")

    documents = ingest_repository(url)

    print(f"\n{'=' * 60}")
    print(f"SAMPLE OUTPUT (first 3 files):")
    print(f"{'=' * 60}")
    for doc in documents[:3]:
        preview = doc["content"][:150].replace("\n", "\\n")
        print(f"\nFile: {doc['file_path']}")
        print(f"Lang: {doc['language']}")
        print(f"Lines: {doc['num_lines']}")
        print(f"Preview: {preview}...")