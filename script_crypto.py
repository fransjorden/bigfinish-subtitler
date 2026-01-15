#!/usr/bin/env python3
"""
Big Finish Script Obfuscation Utility

Obfuscates script files so they're not readable when browsing GitHub,
but the app can still read them automatically.

Usage:
    python script_crypto.py encrypt    # Obfuscate all scripts for GitHub
    python script_crypto.py decrypt    # De-obfuscate scripts (for editing)
    python script_crypto.py test       # Test that it works

The obfuscated files use .enc extension. Delete .json files before pushing to GitHub.
"""

import sys
import json
import base64
import zlib
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "parsed_scripts"

# Obfuscation key - not meant to be secure, just to prevent casual reading
# This is intentionally embedded so the app works without configuration
_OBF_KEY = b"BigFinishSubtitler2024AccessibilityTool"


def _obfuscate(data: bytes) -> bytes:
    """Obfuscate data so it's not human-readable."""
    # Compress first
    compressed = zlib.compress(data, level=9)
    # XOR with repeating key
    key_cycle = (_OBF_KEY * ((len(compressed) // len(_OBF_KEY)) + 1))[:len(compressed)]
    obfuscated = bytes(a ^ b for a, b in zip(compressed, key_cycle))
    # Base64 encode
    return base64.b64encode(obfuscated)


def _deobfuscate(data: bytes) -> bytes:
    """Reverse the obfuscation."""
    # Base64 decode
    obfuscated = base64.b64decode(data)
    # XOR with repeating key
    key_cycle = (_OBF_KEY * ((len(obfuscated) // len(_OBF_KEY)) + 1))[:len(obfuscated)]
    compressed = bytes(a ^ b for a, b in zip(obfuscated, key_cycle))
    # Decompress
    return zlib.decompress(compressed)


def obfuscate_file(filepath: Path) -> bool:
    """Obfuscate a single file."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()

        obfuscated = _obfuscate(data)
        enc_path = filepath.with_suffix('.enc')

        with open(enc_path, 'wb') as f:
            f.write(obfuscated)

        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def deobfuscate_file(filepath: Path) -> Optional[bytes]:
    """Deobfuscate a file and return its contents."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        return _deobfuscate(data)
    except Exception as e:
        print(f"  Error: {e}")
        return None


def encrypt_all():
    """Obfuscate all script JSON files."""
    print("Obfuscating scripts for GitHub...")
    print()

    json_files = list(SCRIPTS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found in parsed_scripts/")
        return

    success = 0
    for filepath in sorted(json_files):
        print(f"  {filepath.name} -> {filepath.stem}.enc")
        if obfuscate_file(filepath):
            success += 1

    print()
    print(f"Done! Obfuscated {success}/{len(json_files)} files")
    print()
    print("Next steps:")
    print("  1. Delete the .json files: del parsed_scripts\\*.json")
    print("  2. Commit and push the .enc files to GitHub")


def decrypt_all():
    """Deobfuscate all encrypted script files."""
    print("Deobfuscating scripts...")
    print()

    enc_files = list(SCRIPTS_DIR.glob("*.enc"))
    if not enc_files:
        print("No .enc files found in parsed_scripts/")
        return

    success = 0
    for filepath in sorted(enc_files):
        print(f"  {filepath.name} -> {filepath.stem}.json")
        data = deobfuscate_file(filepath)
        if data:
            json_path = filepath.with_suffix('.json')
            with open(json_path, 'wb') as f:
                f.write(data)
            success += 1

    print()
    print(f"Done! Deobfuscated {success}/{len(enc_files)} files")


def test():
    """Test obfuscation/deobfuscation."""
    print("Testing obfuscation...")
    print()

    # Find a script to test with
    json_files = list(SCRIPTS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found to test with")
        return

    test_file = json_files[0]
    print(f"Test file: {test_file.name}")

    # Read original
    with open(test_file, 'rb') as f:
        original = f.read()

    # Obfuscate
    obfuscated = _obfuscate(original)
    print(f"Original size: {len(original):,} bytes")
    print(f"Obfuscated size: {len(obfuscated):,} bytes")
    print(f"Compression: {100 - (len(obfuscated) / len(original) * 100):.1f}%")

    # Deobfuscate
    restored = _deobfuscate(obfuscated)

    # Verify
    if original == restored:
        print()
        print("SUCCESS! Obfuscation works correctly.")
    else:
        print()
        print("FAILED: Restored data doesn't match original!")


# === Functions used by the server ===

def load_script(script_id: str) -> Optional[dict]:
    """Load a script by ID, trying .enc first then .json."""
    enc_path = SCRIPTS_DIR / f"{script_id}.enc"
    json_path = SCRIPTS_DIR / f"{script_id}.json"

    # Try obfuscated file first
    if enc_path.exists():
        data = deobfuscate_file(enc_path)
        if data:
            return json.loads(data.decode('utf-8'))

    # Fall back to plain JSON
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def load_search_index() -> Optional[dict]:
    """Load the search index, trying .enc first then .json."""
    enc_path = SCRIPTS_DIR / "search_index.enc"
    json_path = SCRIPTS_DIR / "search_index.json"

    # Try obfuscated file first
    if enc_path.exists():
        data = deobfuscate_file(enc_path)
        if data:
            return json.loads(data.decode('utf-8'))

    # Fall back to plain JSON
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


# Keep old names for compatibility
def load_encrypted_script(script_id: str, password: str = None) -> Optional[dict]:
    return load_script(script_id)

def load_encrypted_search_index(password: str = None) -> Optional[dict]:
    return load_search_index()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python script_crypto.py encrypt   # Obfuscate scripts for GitHub")
        print("  python script_crypto.py decrypt   # Restore JSON files")
        print("  python script_crypto.py test      # Test obfuscation")
        sys.exit(1)

    command = sys.argv[1]

    if command == "encrypt":
        encrypt_all()
    elif command == "decrypt":
        decrypt_all()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
