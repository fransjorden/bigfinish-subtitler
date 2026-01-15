#!/usr/bin/env python3
"""
Big Finish Script Import Utility

Easily import new .txt script files into the system:
1. Parses .txt files into JSON format
2. Rebuilds the search index
3. Makes new stories available in the web app

Usage:
    python import_scripts.py                   # Process all files in raw_scripts/
    python import_scripts.py path/to/file.txt  # Process a specific file
    python import_scripts.py path/to/folder/   # Process all .txt files in folder
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from script_parser import parse_script, asdict
from build_search_index import build_search_index
import json


RAW_SCRIPTS_DIR = PROJECT_ROOT / "raw_scripts"
PARSED_SCRIPTS_DIR = PROJECT_ROOT / "parsed_scripts"
SEARCH_INDEX_PATH = PARSED_SCRIPTS_DIR / "search_index.json"


def ensure_directories():
    """Create necessary directories if they don't exist"""
    RAW_SCRIPTS_DIR.mkdir(exist_ok=True)
    PARSED_SCRIPTS_DIR.mkdir(exist_ok=True)


def parse_single_file(filepath: Path) -> bool:
    """Parse a single .txt file and save to parsed_scripts/"""
    print(f"Parsing: {filepath.name}")

    try:
        script = parse_script(str(filepath))

        # Save to parsed_scripts directory
        output_file = PARSED_SCRIPTS_DIR / f"{script.id}.json"
        script_dict = asdict(script)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(script_dict, f, indent=2, ensure_ascii=False)

        print(f"  -> Created: {output_file.name}")
        print(f"     Title: {script.title}")
        print(f"     Release: #{script.release_number}")
        print(f"     Parts: {len(script.parts)}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def import_scripts(source: Path = None):
    """
    Import scripts from a source path.
    If source is None, processes all files in raw_scripts/
    """
    ensure_directories()

    files_to_process = []

    if source is None:
        # Process raw_scripts directory
        source = RAW_SCRIPTS_DIR
        files_to_process = list(source.glob("*.txt"))
        if not files_to_process:
            print(f"No .txt files found in {RAW_SCRIPTS_DIR}")
            print(f"\nTo add scripts:")
            print(f"  1. Place .txt files in: {RAW_SCRIPTS_DIR}")
            print(f"  2. Run: python import_scripts.py")
            print(f"\nFile naming format: NNN. Title - Author.txt")
            print(f"  Example: 150. The Chimes of Midnight - Robert Shearman.txt")
            return
    elif source.is_file():
        files_to_process = [source]
    elif source.is_dir():
        files_to_process = list(source.glob("*.txt"))
    else:
        print(f"Error: {source} not found")
        return

    if not files_to_process:
        print(f"No .txt files found in {source}")
        return

    print(f"Found {len(files_to_process)} file(s) to process\n")

    # Parse each file
    success_count = 0
    for filepath in sorted(files_to_process):
        # Skip README files
        if filepath.name.lower() == 'readme.txt':
            print(f"  Skipping: {filepath.name}")
            continue
        if parse_single_file(filepath):
            success_count += 1
        print()

    print(f"Parsed {success_count}/{len(files_to_process)} files successfully")

    if success_count > 0:
        # Rebuild search index
        print("\nRebuilding search index...")
        build_search_index(str(PARSED_SCRIPTS_DIR), str(SEARCH_INDEX_PATH))

        print("\nDone! New stories are now available in the web app.")
        print("Restart the server if it's running to see changes.")


def main():
    print("Big Finish Script Import Utility")
    print("=" * 40)
    print()

    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        import_scripts(source)
    else:
        import_scripts()


if __name__ == "__main__":
    main()
