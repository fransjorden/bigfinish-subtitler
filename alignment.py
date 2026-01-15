#!/usr/bin/env python3
"""
Transcript Alignment for Big Finish Caption Sync

Aligns AI-generated transcript with official script text to produce
corrected captions with accurate timestamps.

Uses a modified Smith-Waterman algorithm for local sequence alignment,
adapted for text matching.
"""

import re
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class AlignedCaption:
    """A single caption with timing and text"""
    text: str           # Corrected text from script
    start: float        # Start time in seconds
    end: float          # End time in seconds
    confidence: float   # Alignment confidence (0-1)
    original: str       # Original transcript text (for debugging)
    speaker: str = ""   # Speaker name if available


@dataclass
class ScriptWord:
    """A word from the script with metadata"""
    original: str       # Original text with punctuation/ligatures
    normalized: str     # Lowercase, no punctuation (for matching)
    speaker: str        # Character speaking this word
    element_idx: int    # Index into elements array


@dataclass
class GapSegment:
    """A gap between captions - may be music, recap, or silence"""
    start: float            # Start time in seconds
    end: float              # End time in seconds
    gap_type: str           # "music", "recap", "silence"
    has_speech: bool        # Whether transcript detected speech
    transcript_text: str    # What whisper heard (for debugging/display)


@dataclass
class AlignmentResult:
    """Complete alignment result"""
    captions: list[AlignedCaption]
    gaps: list[GapSegment]  # Gaps between/before/after captions
    script_coverage: float  # What % of script was matched
    transcript_coverage: float  # What % of transcript was used
    average_confidence: float


def normalize_for_alignment(text: str) -> str:
    """Normalize text for alignment comparison"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Split text into words (normalized)"""
    return normalize_for_alignment(text).split()


def tokenize_preserving_original(text: str) -> list[tuple[str, str]]:
    """Split text into (original, normalized) word pairs"""
    # Split on whitespace but keep original words
    words = text.split()
    result = []
    for word in words:
        normalized = normalize_for_alignment(word)
        if normalized:  # Skip if word becomes empty after normalization
            result.append((word, normalized))
    return result


def build_script_words_from_elements(elements: list[dict]) -> list[ScriptWord]:
    """
    Build a list of ScriptWord objects from script elements.
    Preserves original text, speaker attribution, and creates normalized form for matching.
    """
    script_words = []

    for elem_idx, element in enumerate(elements):
        if element.get('type') != 'dialogue':
            continue

        text = element.get('text', '')
        speaker = element.get('character', '')

        # Tokenize while preserving original
        word_pairs = tokenize_preserving_original(text)

        for original, normalized in word_pairs:
            script_words.append(ScriptWord(
                original=original,
                normalized=normalized,
                speaker=speaker,
                element_idx=elem_idx
            ))

    return script_words


def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between two words.
    Returns 1.0 for exact match, partial score for similar words.
    """
    w1 = word1.lower()
    w2 = word2.lower()
    
    if w1 == w2:
        return 1.0
    
    # Check for common transcription errors
    # Numbers written differently
    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    if w1 in number_words and number_words[w1] == w2:
        return 0.9
    if w2 in number_words and number_words[w2] == w1:
        return 0.9
    
    # Levenshtein-based similarity for close matches
    if abs(len(w1) - len(w2)) <= 2:
        distance = levenshtein_distance(w1, w2)
        max_len = max(len(w1), len(w2))
        if distance <= 2:
            return 1.0 - (distance / max_len)
    
    return 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def align_sequences(transcript_words: list[dict],
                   script_words: list,
                   match_score: float = 2.0,
                   mismatch_penalty: float = -1.0,
                   gap_penalty: float = -0.5) -> list[tuple]:
    """
    Align transcript words with script words using dynamic programming.

    Args:
        transcript_words: List of dicts with 'word', 'start', 'end' keys
        script_words: List of words (strings) or ScriptWord objects

    Returns:
        List of (transcript_idx, script_idx, score) alignment pairs
    """
    n = len(transcript_words)
    m = len(script_words)

    if n == 0 or m == 0:
        return []

    # Handle both string list and ScriptWord list
    def get_script_word_normalized(idx):
        sw = script_words[idx]
        if isinstance(sw, ScriptWord):
            return sw.normalized
        return sw.lower()

    # Initialize scoring matrix
    # Using local alignment (Smith-Waterman style)
    score_matrix = [[0.0] * (m + 1) for _ in range(n + 1)]
    traceback = [[None] * (m + 1) for _ in range(n + 1)]

    max_score = 0
    max_pos = (0, 0)

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            t_word = transcript_words[i-1]['word'].lower()
            s_word = get_script_word_normalized(j-1)

            # Calculate match/mismatch score
            similarity = word_similarity(t_word, s_word)
            if similarity > 0.5:
                diag_score = score_matrix[i-1][j-1] + (match_score * similarity)
            else:
                diag_score = score_matrix[i-1][j-1] + mismatch_penalty

            # Gap scores
            up_score = score_matrix[i-1][j] + gap_penalty
            left_score = score_matrix[i][j-1] + gap_penalty

            # Local alignment: can reset to 0
            best_score = max(0, diag_score, up_score, left_score)
            score_matrix[i][j] = best_score

            if best_score == diag_score and diag_score > 0:
                traceback[i][j] = 'diag'
            elif best_score == up_score and up_score > 0:
                traceback[i][j] = 'up'
            elif best_score == left_score and left_score > 0:
                traceback[i][j] = 'left'

            if best_score > max_score:
                max_score = best_score
                max_pos = (i, j)

    # Traceback to find alignment
    alignments = []
    i, j = max_pos

    while i > 0 and j > 0 and traceback[i][j] is not None:
        if traceback[i][j] == 'diag':
            t_word = transcript_words[i-1]['word']
            s_word = get_script_word_normalized(j-1)
            similarity = word_similarity(t_word.lower(), s_word)

            if similarity > 0.5:
                alignments.append((i-1, j-1, similarity))
            i -= 1
            j -= 1
        elif traceback[i][j] == 'up':
            i -= 1
        else:
            j -= 1

    alignments.reverse()
    return alignments


def create_captions_from_alignment(transcript_words: list[dict],
                                   script_text: str,
                                   alignments: list[tuple],
                                   words_per_caption: int = 10,
                                   max_caption_duration: float = 5.0) -> list[AlignedCaption]:
    """
    Create caption entries from alignment data.
    
    Groups aligned words into caption-sized chunks with timing from transcript
    and text from script.
    """
    if not alignments:
        return []
    
    script_words = tokenize(script_text)
    
    # Group alignments into captions
    captions = []
    current_group = []
    current_script_indices = []
    
    for t_idx, s_idx, score in alignments:
        t_word = transcript_words[t_idx]
        s_word = script_words[s_idx] if s_idx < len(script_words) else t_word['word']
        
        current_group.append({
            'transcript': t_word,
            'script_word': s_word,
            'score': score
        })
        current_script_indices.append(s_idx)
        
        # Check if we should start a new caption
        should_break = False
        
        if len(current_group) >= words_per_caption:
            should_break = True
        elif current_group:
            duration = t_word['end'] - current_group[0]['transcript']['start']
            if duration >= max_caption_duration:
                should_break = True
        
        if should_break and current_group:
            # Create caption from current group
            caption = _create_caption_from_group(current_group, current_script_indices, script_words)
            captions.append(caption)
            current_group = []
            current_script_indices = []
    
    # Handle remaining words
    if current_group:
        caption = _create_caption_from_group(current_group, current_script_indices, script_words)
        captions.append(caption)
    
    return captions


def _create_caption_from_group(group: list[dict], 
                               script_indices: list[int],
                               script_words: list[str]) -> AlignedCaption:
    """Create a single caption from a group of aligned words"""
    
    # Get timing from transcript
    start = group[0]['transcript']['start']
    end = group[-1]['transcript']['end']
    
    # Get text from script - use the range of script indices
    if script_indices:
        min_idx = min(script_indices)
        max_idx = max(script_indices)
        # Include words between matched indices for better flow
        script_text_words = script_words[min_idx:max_idx + 1]
        text = ' '.join(script_text_words)
    else:
        text = ' '.join(g['transcript']['word'] for g in group)
    
    # Calculate confidence as average of alignment scores
    avg_confidence = sum(g['score'] for g in group) / len(group)
    
    # Original transcript text for debugging
    original = ' '.join(g['transcript']['word'] for g in group)
    
    return AlignedCaption(
        text=text,
        start=start,
        end=end,
        confidence=avg_confidence,
        original=original
    )


def align_transcript_to_script(transcript_data: dict, 
                               script_dialogue: str) -> AlignmentResult:
    """
    Main alignment function.
    
    Args:
        transcript_data: Transcript JSON with segments and words
        script_dialogue: The searchable_dialogue text from the script
        
    Returns:
        AlignmentResult with captions
    """
    # Extract all words with timestamps from transcript
    transcript_words = []
    for segment in transcript_data.get('segments', []):
        for word in segment.get('words', []):
            transcript_words.append({
                'word': word['word'],
                'start': word['start'],
                'end': word['end']
            })
    
    if not transcript_words:
        return AlignmentResult(
            captions=[],
            gaps=[],
            script_coverage=0.0,
            transcript_coverage=0.0,
            average_confidence=0.0
        )

    # Tokenize script
    script_words = tokenize(script_dialogue)

    # Perform alignment
    alignments = align_sequences(transcript_words, script_words)

    # Create captions
    captions = create_captions_from_alignment(
        transcript_words,
        script_dialogue,
        alignments
    )

    # Detect gaps
    gaps = detect_gaps(transcript_words, captions, alignments)

    # Calculate coverage stats
    if alignments:
        transcript_indices = set(a[0] for a in alignments)
        script_indices = set(a[1] for a in alignments)

        transcript_coverage = len(transcript_indices) / len(transcript_words)
        script_coverage = len(script_indices) / len(script_words) if script_words else 0
        avg_confidence = sum(c.confidence for c in captions) / len(captions) if captions else 0
    else:
        transcript_coverage = 0
        script_coverage = 0
        avg_confidence = 0

    return AlignmentResult(
        captions=captions,
        gaps=gaps,
        script_coverage=script_coverage,
        transcript_coverage=transcript_coverage,
        average_confidence=avg_confidence
    )


def align_transcript_to_script_elements(transcript_data: dict,
                                         elements: list[dict]) -> AlignmentResult:
    """
    Align transcript to script using structured elements.
    Preserves original text formatting and speaker attribution.

    Args:
        transcript_data: Transcript JSON with segments and words
        elements: Script elements with 'type', 'text', 'character' fields

    Returns:
        AlignmentResult with captions including speaker names
    """
    # Extract all words with timestamps from transcript
    transcript_words = []
    for segment in transcript_data.get('segments', []):
        for word in segment.get('words', []):
            transcript_words.append({
                'word': word['word'],
                'start': word['start'],
                'end': word['end']
            })

    if not transcript_words:
        return AlignmentResult(
            captions=[],
            gaps=[],
            script_coverage=0.0,
            transcript_coverage=0.0,
            average_confidence=0.0
        )

    # Build script words with speaker info
    script_words = build_script_words_from_elements(elements)

    if not script_words:
        return AlignmentResult(
            captions=[],
            gaps=[],
            script_coverage=0.0,
            transcript_coverage=0.0,
            average_confidence=0.0
        )

    # Perform alignment
    alignments = align_sequences(transcript_words, script_words)

    # Create captions with speaker attribution
    captions = create_captions_with_speakers(
        transcript_words,
        script_words,
        alignments
    )

    # Detect gaps (unmatched transcript sections)
    gaps = detect_gaps(transcript_words, captions, alignments)

    # Calculate coverage stats
    if alignments:
        transcript_indices = set(a[0] for a in alignments)
        script_indices = set(a[1] for a in alignments)

        transcript_coverage = len(transcript_indices) / len(transcript_words)
        script_coverage = len(script_indices) / len(script_words)
        avg_confidence = sum(c.confidence for c in captions) / len(captions) if captions else 0
    else:
        transcript_coverage = 0
        script_coverage = 0
        avg_confidence = 0

    return AlignmentResult(
        captions=captions,
        gaps=gaps,
        script_coverage=script_coverage,
        transcript_coverage=transcript_coverage,
        average_confidence=avg_confidence
    )


def create_captions_with_speakers(transcript_words: list[dict],
                                   script_words: list[ScriptWord],
                                   alignments: list[tuple],
                                   words_per_caption: int = 10,
                                   max_caption_duration: float = 5.0) -> list[AlignedCaption]:
    """
    Create captions from alignment, preserving original text and speaker names.
    """
    if not alignments:
        return []

    captions = []
    current_group = []
    current_speaker = None

    for t_idx, s_idx, score in alignments:
        t_word = transcript_words[t_idx]
        s_word = script_words[s_idx]

        # Check if speaker changed - force new caption
        speaker_changed = current_speaker is not None and s_word.speaker != current_speaker

        current_group.append({
            'transcript': t_word,
            'script_word': s_word,
            'score': score
        })

        # Determine if we should break to new caption
        should_break = False

        if speaker_changed:
            # Don't include this word in current group - start fresh
            word_to_move = current_group.pop()
            if current_group:
                caption = _create_caption_with_speaker(current_group)
                captions.append(caption)
            current_group = [word_to_move]
            current_speaker = s_word.speaker
            continue

        if len(current_group) >= words_per_caption:
            should_break = True
        elif current_group:
            duration = t_word['end'] - current_group[0]['transcript']['start']
            if duration >= max_caption_duration:
                should_break = True

        if should_break and current_group:
            caption = _create_caption_with_speaker(current_group)
            captions.append(caption)
            current_group = []

        current_speaker = s_word.speaker

    # Handle remaining words
    if current_group:
        caption = _create_caption_with_speaker(current_group)
        captions.append(caption)

    return captions


def _create_caption_with_speaker(group: list[dict]) -> AlignedCaption:
    """Create a caption preserving original text and speaker."""
    # Get timing from transcript
    start = group[0]['transcript']['start']
    end = group[-1]['transcript']['end']

    # Get speaker from first word (should be same for whole group)
    speaker = group[0]['script_word'].speaker

    # Build text from original script words
    text_words = [g['script_word'].original for g in group]
    text = ' '.join(text_words)

    # Add speaker prefix
    if speaker:
        text = f"{speaker}: {text}"

    # Calculate confidence
    avg_confidence = sum(g['score'] for g in group) / len(group)

    # Original transcript text
    original = ' '.join(g['transcript']['word'] for g in group)

    return AlignedCaption(
        text=text,
        start=start,
        end=end,
        confidence=avg_confidence,
        original=original,
        speaker=speaker
    )


def detect_gaps(transcript_words: list[dict],
                captions: list[AlignedCaption],
                alignments: list[tuple],
                min_gap_duration: float = 3.0) -> list[GapSegment]:
    """
    Detect gaps in the captions and classify them.

    Gap types:
    - "music": No speech detected (intro/outro music)
    - "recap": Speech detected but doesn't match script (previous episode recap)
    - "silence": Short gap between dialogue

    Args:
        transcript_words: All words from transcript with timestamps
        captions: Generated captions
        alignments: Alignment pairs (transcript_idx, script_idx, score)
        min_gap_duration: Minimum gap duration to report (seconds)

    Returns:
        List of GapSegment objects
    """
    if not transcript_words:
        return []

    gaps = []
    matched_transcript_indices = set(a[0] for a in alignments)

    # Get audio duration from last transcript word
    audio_duration = transcript_words[-1]['end'] if transcript_words else 0

    # Find all time ranges and their status
    # First, identify unmatched transcript regions
    unmatched_regions = []
    current_region = None

    for i, tw in enumerate(transcript_words):
        is_matched = i in matched_transcript_indices

        if not is_matched:
            if current_region is None:
                current_region = {
                    'start': tw['start'],
                    'end': tw['end'],
                    'words': [tw['word']]
                }
            else:
                current_region['end'] = tw['end']
                current_region['words'].append(tw['word'])
        else:
            if current_region is not None:
                unmatched_regions.append(current_region)
                current_region = None

    if current_region is not None:
        unmatched_regions.append(current_region)

    # Convert unmatched regions to gap segments
    for region in unmatched_regions:
        duration = region['end'] - region['start']
        if duration < min_gap_duration:
            continue

        # Has speech if there are words
        has_speech = len(region['words']) > 0
        transcript_text = ' '.join(region['words'])

        # Classify the gap
        if not has_speech:
            gap_type = "music"
        else:
            # Speech that doesn't match script = recap
            gap_type = "recap"

        gaps.append(GapSegment(
            start=region['start'],
            end=region['end'],
            gap_type=gap_type,
            has_speech=has_speech,
            transcript_text=transcript_text
        ))

    # Also detect gaps BEFORE first caption and AFTER last caption
    if captions:
        first_caption_start = captions[0].start
        last_caption_end = captions[-1].end

        # Gap before first caption (intro)
        if first_caption_start > min_gap_duration:
            # Check if there's unmatched speech in this region
            intro_words = [tw for tw in transcript_words if tw['end'] <= first_caption_start]
            intro_text = ' '.join(tw['word'] for tw in intro_words)
            has_intro_speech = len(intro_words) > 5  # More than a few words

            # Check if this intro gap is already covered by unmatched regions
            intro_covered = any(g.start < min_gap_duration and g.end >= first_caption_start - 1
                               for g in gaps)

            if not intro_covered:
                gap_type = "recap" if has_intro_speech else "music"
                gaps.insert(0, GapSegment(
                    start=0,
                    end=first_caption_start,
                    gap_type=gap_type,
                    has_speech=has_intro_speech,
                    transcript_text=intro_text[:200] if intro_text else ""
                ))

        # Gap after last caption (outro)
        if audio_duration - last_caption_end > min_gap_duration:
            outro_words = [tw for tw in transcript_words if tw['start'] >= last_caption_end]
            outro_text = ' '.join(tw['word'] for tw in outro_words)
            has_outro_speech = len(outro_words) > 5

            gaps.append(GapSegment(
                start=last_caption_end,
                end=audio_duration,
                gap_type="music" if not has_outro_speech else "recap",
                has_speech=has_outro_speech,
                transcript_text=outro_text[:200] if outro_text else ""
            ))

    # Sort by start time
    gaps.sort(key=lambda g: g.start)

    return gaps


def export_to_webvtt(captions: list[AlignedCaption], output_path: str):
    """Export captions to WebVTT format"""
    
    def format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    lines = ["WEBVTT", ""]
    
    for i, caption in enumerate(captions, 1):
        lines.append(str(i))
        lines.append(f"{format_time(caption.start)} --> {format_time(caption.end)}")
        lines.append(caption.text)
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_to_json(alignment_result: AlignmentResult, output_path: str):
    """Export alignment result to JSON"""
    data = {
        'captions': [
            {
                'text': c.text,
                'start': c.start,
                'end': c.end,
                'confidence': c.confidence,
                'speaker': c.speaker
            }
            for c in alignment_result.captions
        ],
        'gaps': [
            {
                'start': g.start,
                'end': g.end,
                'type': g.gap_type,
                'has_speech': g.has_speech,
                'transcript_text': g.transcript_text
            }
            for g in alignment_result.gaps
        ],
        'stats': {
            'script_coverage': alignment_result.script_coverage,
            'transcript_coverage': alignment_result.transcript_coverage,
            'average_confidence': alignment_result.average_confidence
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python alignment.py <transcript_json> <script_dialogue_file> <output_vtt>")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    script_file = sys.argv[2]
    output_file = sys.argv[3]
    
    with open(transcript_file) as f:
        transcript_data = json.load(f)
    
    with open(script_file) as f:
        script_dialogue = f.read()
    
    print("Aligning transcript to script...")
    result = align_transcript_to_script(transcript_data, script_dialogue)
    
    print(f"Created {len(result.captions)} captions")
    print(f"Script coverage: {result.script_coverage:.1%}")
    print(f"Transcript coverage: {result.transcript_coverage:.1%}")
    print(f"Average confidence: {result.average_confidence:.1%}")
    
    export_to_webvtt(result.captions, output_file)
    print(f"Saved to: {output_file}")
