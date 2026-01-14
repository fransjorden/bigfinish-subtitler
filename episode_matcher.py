#!/usr/bin/env python3
"""
Episode Matcher for Big Finish Caption Sync

Identifies which script (and which part) matches the uploaded audio
by comparing AI transcription against the script database.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import Counter


@dataclass
class MatchResult:
    """Result of episode matching"""
    script_id: str
    title: str
    part_number: int
    release_number: int
    confidence: float
    runner_up: Optional[dict] = None


def normalize_text(text: str) -> str:
    """Normalize text for matching - lowercase, remove punctuation, collapse whitespace"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_ngrams(text: str, n: int = 3) -> list:
    """Extract word n-grams from text as a list (preserves duplicates for counting)"""
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams


def calculate_similarity(query_text: str, script_text: str) -> float:
    """
    Calculate similarity between query transcript and script text.
    Uses a combination of:
    - Word overlap (Jaccard-like)
    - N-gram overlap
    - Sequence matching bonus for contiguous matches
    """
    query_norm = normalize_text(query_text)
    script_norm = normalize_text(script_text)
    
    query_words = query_norm.split()
    script_words = script_norm.split()
    
    if not query_words or not script_words:
        return 0.0
    
    # Word frequency overlap
    query_freq = Counter(query_words)
    script_freq = Counter(script_words)
    
    # Calculate overlap using minimum of frequencies
    common_words = set(query_freq.keys()) & set(script_freq.keys())
    overlap_count = sum(min(query_freq[w], script_freq[w]) for w in common_words)
    word_score = overlap_count / len(query_words)
    
    # N-gram overlap (trigrams)
    query_ngrams = extract_ngrams(query_norm, 3)
    script_ngrams = set(extract_ngrams(script_norm, 3))
    
    if query_ngrams:
        ngram_matches = sum(1 for ng in query_ngrams if ng in script_ngrams)
        ngram_score = ngram_matches / len(query_ngrams)
    else:
        ngram_score = 0.0
    
    # 4-gram overlap (more specific matching)
    query_4grams = extract_ngrams(query_norm, 4)
    script_4grams = set(extract_ngrams(script_norm, 4))
    
    if query_4grams:
        fourgram_matches = sum(1 for ng in query_4grams if ng in script_4grams)
        fourgram_score = fourgram_matches / len(query_4grams)
    else:
        fourgram_score = 0.0
    
    # Combined score with weights
    # Higher weight on n-grams as they capture sequences better
    score = (word_score * 0.2) + (ngram_score * 0.4) + (fourgram_score * 0.4)
    
    return score


def match_episode(transcript_text: str, 
                  search_index_path: str,
                  scripts_dir: str,
                  top_k: int = 5) -> MatchResult:
    """
    Match transcribed audio to a script episode.
    
    Args:
        transcript_text: Text from AI transcription (first few minutes)
        search_index_path: Path to search_index.json
        scripts_dir: Path to directory with parsed script JSONs
        top_k: Number of candidates to consider
        
    Returns:
        MatchResult with identified episode and confidence
    """
    # Load search index
    with open(search_index_path) as f:
        index = json.load(f)
    
    # First pass: quick search using index samples
    candidates = []
    
    query_norm = normalize_text(transcript_text)
    query_words = set(query_norm.split())
    query_ngrams = set(extract_ngrams(query_norm, 3))
    
    for part in index['parts_index']:
        # Quick score using indexed samples
        sample_words = set(part['sample'].split())
        opening_words = set(part['opening'].split())
        
        word_overlap = len(query_words & (sample_words | opening_words))
        
        sample_ngrams = set(extract_ngrams(part['opening'], 3))
        ngram_overlap = len(query_ngrams & sample_ngrams)
        
        quick_score = word_overlap + (ngram_overlap * 3)  # Weight n-grams higher
        
        if quick_score > 5:  # Minimum threshold
            candidates.append({
                'script_id': part['script_id'],
                'title': part['title'],
                'part_number': part['part_number'],
                'release_number': part['release_number'],
                'quick_score': quick_score
            })
    
    # Sort by quick score and take top candidates
    candidates.sort(key=lambda x: x['quick_score'], reverse=True)
    top_candidates = candidates[:top_k * 2]  # Get more for second pass
    
    if not top_candidates:
        # No matches found
        return MatchResult(
            script_id="unknown",
            title="Unknown Episode",
            part_number=0,
            release_number=0,
            confidence=0.0
        )
    
    # Second pass: detailed matching against full script text
    detailed_scores = []
    scripts_dir = Path(scripts_dir)
    
    for candidate in top_candidates:
        script_path = scripts_dir / f"{candidate['script_id']}.json"
        
        if not script_path.exists():
            continue
        
        with open(script_path) as f:
            script_data = json.load(f)
        
        # Find the right part
        for part in script_data['parts']:
            if part['part_number'] == candidate['part_number']:
                # Get the searchable dialogue for this part
                script_text = part['searchable_dialogue']
                
                # Use first portion of script (corresponding to transcript length)
                # Estimate: transcript is ~3 min, script might be 20-30 min per part
                # So use first ~15% of script text
                script_sample = script_text[:len(transcript_text) * 3]
                
                # Calculate detailed similarity
                score = calculate_similarity(transcript_text, script_sample)
                
                detailed_scores.append({
                    'script_id': candidate['script_id'],
                    'title': candidate['title'],
                    'part_number': candidate['part_number'],
                    'release_number': candidate['release_number'],
                    'score': score
                })
                break
    
    # Sort by detailed score
    detailed_scores.sort(key=lambda x: x['score'], reverse=True)
    
    if not detailed_scores:
        return MatchResult(
            script_id="unknown",
            title="Unknown Episode",
            part_number=0,
            release_number=0,
            confidence=0.0
        )
    
    best = detailed_scores[0]
    runner_up = detailed_scores[1] if len(detailed_scores) > 1 else None
    
    # Calculate confidence based on score and gap to runner-up
    confidence = best['score']
    if runner_up:
        gap = best['score'] - runner_up['score']
        # Boost confidence if there's a clear winner
        if gap > 0.1:
            confidence = min(confidence + 0.1, 1.0)
    
    return MatchResult(
        script_id=best['script_id'],
        title=best['title'],
        part_number=best['part_number'],
        release_number=best['release_number'],
        confidence=confidence,
        runner_up=runner_up
    )


def get_script_part_text(script_id: str, part_number: int, scripts_dir: str) -> Optional[str]:
    """Get the full dialogue text for a specific script part"""
    script_path = Path(scripts_dir) / f"{script_id}.json"
    
    if not script_path.exists():
        return None
    
    with open(script_path) as f:
        script_data = json.load(f)
    
    for part in script_data['parts']:
        if part['part_number'] == part_number:
            return part['searchable_dialogue']
    
    return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python episode_matcher.py <transcript_text_file> <search_index> <scripts_dir>")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    search_index = sys.argv[2]
    scripts_dir = sys.argv[3]
    
    with open(transcript_file) as f:
        transcript_text = f.read()
    
    print(f"Matching transcript ({len(transcript_text)} chars)...")
    
    result = match_episode(transcript_text, search_index, scripts_dir)
    
    print()
    print(f"=== MATCH RESULT ===")
    print(f"Episode: {result.title}")
    print(f"Part: {result.part_number}")
    print(f"Release #: {result.release_number}")
    print(f"Confidence: {result.confidence:.1%}")
    
    if result.runner_up:
        print()
        print(f"Runner-up: {result.runner_up['title']} Part {result.runner_up['part_number']} ({result.runner_up['score']:.1%})")
