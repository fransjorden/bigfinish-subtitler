#!/usr/bin/env python3
"""
Build a search index for episode matching.

Creates a JSON file with:
- All scripts indexed by release number
- Searchable text per part
- Fingerprint data for quick matching
"""

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict


def normalize_text(text: str) -> str:
    """Normalize text for matching - lowercase, remove punctuation, collapse whitespace"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_ngrams(text: str, n: int = 3) -> set:
    """Extract word n-grams from text"""
    words = text.split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams


def create_fingerprint(text: str, num_chunks: int = 10) -> list:
    """
    Create a fingerprint by hashing chunks of the text.
    Useful for quick similarity comparison.
    """
    normalized = normalize_text(text)
    words = normalized.split()
    
    if len(words) < num_chunks * 10:
        # Short text - just hash the whole thing
        return [hashlib.md5(normalized.encode()).hexdigest()[:8]]
    
    chunk_size = len(words) // num_chunks
    fingerprints = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        fp = hashlib.md5(chunk.encode()).hexdigest()[:8]
        fingerprints.append(fp)
    
    return fingerprints


def build_search_index(parsed_dir: str, output_file: str):
    """Build search index from parsed scripts"""
    parsed_path = Path(parsed_dir)
    
    index = {
        'version': 1,
        'scripts': {},
        'parts_index': [],  # Flat list of all parts for searching
        'stats': {
            'total_scripts': 0,
            'total_parts': 0,
            'total_words': 0
        }
    }
    
    # Process each script
    for script_file in sorted(parsed_path.glob('*.json')):
        if script_file.name in ('_index.json', 'search_index.json'):
            continue

        with open(script_file, encoding='utf-8') as f:
            script = json.load(f)
        
        script_id = script['id']
        release_num = script['release_number']
        
        # Store script metadata
        index['scripts'][script_id] = {
            'id': script_id,
            'release_number': release_num,
            'title': script['title'],
            'author': script['author'],
            'filename': script['filename'],
            'num_parts': len(script['parts']),
            'parts': []
        }
        
        # Process each part
        for part in script['parts']:
            part_num = part['part_number']
            dialogue = part['searchable_dialogue']
            
            # Normalize for searching
            normalized = normalize_text(dialogue)
            words = normalized.split()
            word_count = len(words)
            
            # Create fingerprint
            fingerprint = create_fingerprint(dialogue)
            
            # Extract some distinctive phrases (first 500 chars worth)
            sample_text = normalized[:500]
            
            # Extract opening lines (for quick identification)
            opening = normalize_text(dialogue[:200])
            
            part_entry = {
                'script_id': script_id,
                'release_number': release_num,
                'title': script['title'],
                'part_number': part_num,
                'word_count': word_count,
                'fingerprint': fingerprint,
                'opening': opening,
                'sample': sample_text,
            }
            
            index['scripts'][script_id]['parts'].append({
                'part_number': part_num,
                'word_count': word_count,
                'fingerprint': fingerprint
            })
            
            index['parts_index'].append(part_entry)
            
            index['stats']['total_words'] += word_count
        
        index['stats']['total_scripts'] += 1
        index['stats']['total_parts'] += len(script['parts'])
    
    # Save index
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"Search index created: {output_file}")
    print(f"  Scripts: {index['stats']['total_scripts']}")
    print(f"  Parts: {index['stats']['total_parts']}")
    print(f"  Total words: {index['stats']['total_words']:,}")
    
    return index


def search_by_text(index: dict, query_text: str, top_k: int = 5) -> list:
    """
    Search for matching scripts/parts by text similarity.
    Returns top_k matches with scores.
    """
    query_normalized = normalize_text(query_text)
    query_words = set(query_normalized.split())
    query_ngrams = extract_ngrams(query_normalized)
    
    results = []
    
    for part in index['parts_index']:
        # Get sample text and compute word overlap
        sample_words = set(part['sample'].split())
        opening_words = set(part['opening'].split())
        
        # Word overlap score
        word_overlap = len(query_words & sample_words) / max(len(query_words), 1)
        
        # N-gram overlap with opening
        sample_ngrams = extract_ngrams(part['opening'])
        ngram_overlap = len(query_ngrams & sample_ngrams) / max(len(query_ngrams), 1)
        
        # Combined score
        score = word_overlap * 0.4 + ngram_overlap * 0.6
        
        if score > 0.1:  # Minimum threshold
            results.append({
                'script_id': part['script_id'],
                'title': part['title'],
                'part_number': part['part_number'],
                'release_number': part['release_number'],
                'score': score
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python build_search_index.py <parsed_dir> <output_file>")
        sys.exit(1)
    
    parsed_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    build_search_index(parsed_dir, output_file)
