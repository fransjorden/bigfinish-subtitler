#!/usr/bin/env python3
"""
Fix newly parsed scripts to have correct titles and IDs.
"""

import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "parsed_scripts"

# Mapping from wrong ID patterns to correct info
FIXES = {
    # (wrong_pattern, correct_id, correct_title, correct_release)
    "028-opening-doctor-who-theme-tune-arranged": ("028-invaders-from-mars", "Invaders from Mars", 28),
    "035-sometimes-hyphenated-or-preceded": ("035-ish", "...ish", 35),
    "062-grown-ever-more-menacing-and-which-i-fear-cannot-now-be-lifted": ("062-the-last", "The Last", 62),
    "064-lamp-light-yes-yes": ("064-the-next-life", "The Next Life", 64),
    "077-designed": ("077-other-lives", "Other Lives", 77),
    "093-renaissance-of-the-daleks-from-a-story": ("093-renaissance-of-the-daleks", "Renaissance of the Daleks", 93),
    "100-100-bc": ("100-100", "100", 100),
    "101-c-rizz-absolution-noun-the-remission-of-sins-granted": ("101-absolution", "Absolution", 101),
    "010-10da1x1": ("000-10da1x1", "Technophobia", 0),
    "010-10da1x2": ("000-10da1x2", "Time Reaver", 0),
    "010-10da1x3": ("000-10da1x3", "Death and the Queen", 0),
    "115-false-gods": ("115-forty-five", "Forty-Five", 115),
    "117-the-key-2-time-the-judgement-of-isskar": ("117-the-judgement-of-isskar", "The Judgement of Isskar", 117),
    "119-the-key-2-time-the-chaos-pool": ("119-the-chaos-pool", "The Chaos Pool", 119),
    "236-donc-iloik": ("236-serpent-in-the-silver-mask", "Serpent in the Silver Mask", 236),
    "244-by-steve-lyons": ("244-warlocks-cross", "Warlock's Cross", 244),
    "250-bi-bic": ("250-the-monsters-of-gokroth", "The Monsters of Gokroth", 250),
    "251-bi-bic": ("251-the-moons-of-vulpana", "The Moons of Vulpana", 251),
    "252-bi-bic": ("252-an-alien-werewolf-in-london", "An Alien Werewolf in London", 252),
    "255-a-four-part-story": ("255-harry-houdinis-war", "Harry Houdini's War", 255),
    "256-the-doctor-peter-davison": ("256-tartarus", "Tartarus", 256),
    "257-iterstitial": ("257-interstitial-feast-of-fear", "Interstitial / Feast of Fear", 257),
    "258-the-doctor": ("258-warzone-conversion", "Warzone / Conversion", 258),
    "259-and-other-stories": ("259-blood-on-santas-claw", "Blood on Santa's Claw and Other Stories", 259),
    "261-bi-bic": ("261-the-psychic-circus", "The Psychic Circus", 261),
}


def fix_scripts():
    """Fix script files with wrong IDs/titles."""
    print("Fixing newly parsed scripts...")
    print()

    fixed = 0

    for wrong_id, (correct_id, correct_title, correct_release) in FIXES.items():
        wrong_path = SCRIPTS_DIR / f"{wrong_id}.json"
        correct_path = SCRIPTS_DIR / f"{correct_id}.json"

        if not wrong_path.exists():
            continue

        print(f"  {wrong_id} -> {correct_id}")

        # Load the script
        with open(wrong_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Fix the metadata
        data['id'] = correct_id
        data['title'] = correct_title
        data['release_number'] = correct_release

        # Save with correct filename
        with open(correct_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Delete old file
        wrong_path.unlink()
        fixed += 1

    print()
    print(f"Fixed {fixed} scripts")
    return fixed


def rebuild_search_index():
    """Rebuild the search index from all script files."""
    print()
    print("Rebuilding search index...")

    index = {
        "version": 1,
        "scripts": {}
    }

    # Find all script JSON files (excluding _index.json and search_index.json)
    script_files = [f for f in SCRIPTS_DIR.glob("*.json")
                    if f.name not in ('_index.json', 'search_index.json')]

    for filepath in sorted(script_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            script_id = data['id']

            # Build fingerprint from parts
            parts_info = []
            for part in data.get('parts', []):
                # Generate fingerprint from searchable dialogue
                dialogue = part.get('searchable_dialogue', '')
                words = dialogue.split()
                word_count = len(words)

                # Simple hash-based fingerprint
                import hashlib
                fingerprint = []
                chunk_size = max(1, word_count // 10)
                for i in range(10):
                    start = i * chunk_size
                    end = start + chunk_size
                    chunk = ' '.join(words[start:end])
                    h = hashlib.md5(chunk.encode()).hexdigest()[:8]
                    fingerprint.append(h)

                parts_info.append({
                    'part_number': part['part_number'],
                    'word_count': word_count,
                    'fingerprint': fingerprint
                })

            index['scripts'][script_id] = {
                'id': script_id,
                'release_number': data['release_number'],
                'title': data['title'],
                'author': data.get('author', ''),
                'filename': data.get('filename', ''),
                'num_parts': len(parts_info),
                'parts': parts_info
            }

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    # Save the index
    index_path = SCRIPTS_DIR / "search_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Search index rebuilt with {len(index['scripts'])} scripts")


def main():
    print("=" * 60)
    print("Fix New Scripts")
    print("=" * 60)
    print()

    # Step 1: Fix script files
    fix_scripts()

    # Step 2: Rebuild search index
    rebuild_search_index()

    print()
    print("=" * 60)
    print("Done! Now run: python script_crypto.py encrypt")
    print("=" * 60)


if __name__ == "__main__":
    main()
