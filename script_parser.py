#!/usr/bin/env python3
"""
Big Finish Doctor Who Script Parser

Parses OCR-scanned scripts into structured JSON format with:
- Preamble detection and skipping
- Part boundary detection
- Dialogue extraction
- Searchable text generation
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class ElementType(str, Enum):
    DIALOGUE = "dialogue"
    DIRECTION = "direction"
    SOUND = "sound"
    SCENE_HEADING = "scene_heading"


@dataclass
class ScriptElement:
    type: ElementType
    text: str
    character: Optional[str] = None


@dataclass
class ScriptPart:
    part_number: int
    start_line: int
    end_line: int
    elements: list = field(default_factory=list)
    searchable_dialogue: str = ""


@dataclass
class ParsedScript:
    id: str
    release_number: int
    title: str
    author: str
    filename: str
    format_type: str
    preamble_end_line: int
    parts: list = field(default_factory=list)
    cast_list: list = field(default_factory=list)
    parse_warnings: list = field(default_factory=list)


# Common character names in Doctor Who (helps identify dialogue)
KNOWN_CHARACTERS = {
    "DOCTOR", "TARDIS", "CHARLEY", "EVELYN", "PERI", "MEL", "ACE", "HEX",
    "BRIGADIER", "ROMANA", "LEELA", "NYSSA", "TEGAN", "TURLOUGH", "ADRIC",
    "SARAH", "JO", "JAMIE", "ZOE", "VICTORIA", "BEN", "POLLY", "STEVEN",
    "VICKI", "IAN", "BARBARA", "SUSAN", "LUCIE", "C'RIZZ", "ERIMEM",
    "FLIP", "CONSTANCE", "MRS WIBBSEY", "MIKE", "BENNY", "NARRATOR",
    "DALEK", "DALEKS", "CYBERMAN", "CYBERMEN", "MASTER", "DAVROS",
    "ANNOUNCER", "TANNOY", "COMPUTER", "VOICE", "MAN", "WOMAN",
    "PRESIDENT", "CAPTAIN", "SERGEANT", "CORPORAL", "LIEUTENANT",
    "VANSELL", "RASSILON", "BORUSA", "CASTELLAN"
}


def extract_release_number(filename: str) -> int:
    """Extract the release number from filename like '001. The Sirens...' or '009 - The Spectre...'"""
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def extract_title_from_filename(filename: str) -> tuple[str, str]:
    """Extract title and author from filename"""
    # Remove number prefix and extension
    name = re.sub(r'^\d+[\.\s\-]+', '', filename)
    name = re.sub(r'_djvu\.txt$', '', name)
    
    title = name
    author = ""
    
    # Extract author if present (after ' - ')
    if ' - ' in name:
        parts = name.split(' - ')
        if len(parts) >= 2:
            title = parts[0].strip()
            author = parts[-1].strip()  # Last part is usually author
    
    return title.strip(), author.strip()


def clean_ocr_text(text: str) -> str:
    """Fix common OCR errors"""
    # Fix | used instead of I
    text = re.sub(r'\b\|\b', 'I', text)
    text = re.sub(r'\|(?=[a-z])', 'I', text)  # |'m -> I'm
    text = re.sub(r'(?<=[a-z])\|', 'I', text)  # we|l -> well
    
    # Fix common OCR substitutions
    text = text.replace('|', 'I')  # Catch remaining pipes that should be I
    
    return text


def find_preamble_end(lines: list[str]) -> int:
    """
    Find where the actual script begins (after any preamble/essay).
    Returns the line number where the script starts.
    """
    # Patterns that indicate script has started
    script_start_patterns = [
        r'^\[Part (One|Two|Three|Four|1|2|3|4)\]',  # [Part One]
        r'^PART (ONE|TWO|THREE|FOUR|1|2|3|4)\s*$',  # PART ONE
        r'^Part (One|Two|Three|Four|1|2|3|4):?\s*$',  # Part One:
        r'^SCENE\s*\d*:',  # SCENE 1:
        r'^\d+:\s*(INT|EXT|INT/EXT)',  # 1: INT. TARDIS
        r'^CAST\s*$',  # CAST list header
        r'^THE DOCTOR:',  # Specific character speaking
        r'^DOCTOR:',
    ]
    
    # Also look for a cast list pattern
    cast_list_pattern = r'^[A-Z][A-Z\s]+:\s*[A-Z][a-z]+'  # CHARACTER: Actor Name
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check for script start patterns
        for pattern in script_start_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                return i
        
        # Check for cast list (multiple lines of CHARACTER: Actor)
        if re.match(cast_list_pattern, stripped):
            # Verify it's actually a cast list by checking next few lines
            cast_count = 0
            for j in range(i, min(i + 10, len(lines))):
                if re.match(cast_list_pattern, lines[j].strip()):
                    cast_count += 1
            if cast_count >= 3:  # At least 3 cast entries
                return i
    
    # If no clear start found, look for first dialogue
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for CHARACTER: dialogue or CHARACTER (direction) dialogue
        if re.match(r'^[A-Z][A-Z\s]{2,20}:', stripped):
            return max(0, i - 5)  # Start a few lines before
    
    return 0


def find_part_boundaries(lines: list[str], start_line: int) -> list[tuple[int, int, int]]:
    """
    Find part boundaries in the script.
    Returns list of (part_number, start_line, end_line) tuples.
    """
    part_markers = []
    
    # Patterns for part markers
    part_patterns = [
        (r'^\[Part (One|Two|Three|Four)\]', {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4}),
        (r'^\[Part ([1-4])\]', None),
        (r'^PART (ONE|TWO|THREE|FOUR)\s*$', {'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4}),
        (r'^PART ([1-4])\s*$', None),
        (r'^Part (One|Two|Three|Four):?\s*$', {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4}),
        (r'^EPISODE\s*([1-4])\s*$', None),
    ]
    
    for i in range(start_line, len(lines)):
        stripped = lines[i].strip()
        
        for pattern, mapping in part_patterns:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                part_val = match.group(1)
                if mapping:
                    part_num = mapping.get(part_val, mapping.get(part_val.upper(), 0))
                else:
                    part_num = int(part_val)
                
                if part_num > 0:
                    part_markers.append((part_num, i))
                break
    
    # Convert to boundaries
    boundaries = []
    seen_parts = set()
    
    for idx, (part_num, start) in enumerate(part_markers):
        # Skip duplicate part numbers (can happen in scripts with excerpts)
        if part_num in seen_parts:
            continue
        seen_parts.add(part_num)
        
        # Find end: next part marker or end of file
        end = len(lines) - 1
        for next_idx in range(idx + 1, len(part_markers)):
            next_part_num, next_start = part_markers[next_idx]
            if next_part_num not in seen_parts or next_part_num > part_num:
                end = next_start - 1
                break
        
        boundaries.append((part_num, start, end))
    
    # If no parts found, treat entire script as Part 1
    if not boundaries:
        boundaries = [(1, start_line, len(lines) - 1)]
    
    return boundaries


def is_character_line(line: str) -> tuple[Optional[str], Optional[str]]:
    """
    Check if line starts with a character name.
    Returns (character_name, dialogue_on_same_line) tuple.
    dialogue_on_same_line is None if character name is alone on line.
    """
    stripped = line.strip()
    
    # Pattern 1: CHARACTER: dialogue or CHARACTER (direction): dialogue
    match = re.match(r'^([A-Z][A-Z\s\'\.]{1,30}?)(?:\s*\([^)]+\))?\s*:\s*(.*)$', stripped)
    if match:
        char = match.group(1).strip()
        dialogue = match.group(2).strip() if match.group(2) else None
        # Filter out non-character patterns
        if char and not re.match(r'^(SCENE|PART|EPISODE|INT|EXT|F/X|SFX|NOTE|NB)', char):
            return char, dialogue
    
    # Pattern 2: CHARACTER dialogue (no colon, screenplay format)
    # Match CHARACTER (all caps, 3-20 chars) followed by space and dialogue
    match = re.match(r'^([A-Z][A-Z]{2,20})\s+(.+)$', stripped)
    if match:
        char = match.group(1).strip()
        dialogue = match.group(2).strip()
        # Verify it looks like a character name (not a direction word)
        if not re.match(r'^(SCENE|PART|EPISODE|INT|EXT|INTERIOR|EXTERIOR|THE|AND|BUT|WITH|LATER|CONTINUED|CUT|FADE|MUSIC|GRAMS|CAPTION)', char):
            # Check dialogue doesn't start with a preposition/article (likely a direction)
            if not re.match(r'^(of|in|on|at|to|from|with|and|the|a|an)\s', dialogue.lower()):
                return char, dialogue
    
    # Pattern 3: CHARACTER alone on line (followed by dialogue on next line)
    if re.match(r'^[A-Z][A-Z\s]{2,25}$', stripped):
        # Check it's not a scene heading or direction
        if stripped in KNOWN_CHARACTERS or not re.match(r'^(THE|A|AN|IN|ON|AT|BY)\s', stripped):
            if not re.match(r'^(SCENE|PART|EPISODE|INT|EXT|INTERIOR|EXTERIOR|CONTINUED|END|CUT|FADE)', stripped):
                return stripped, None
    
    return None, None


def is_stage_direction(line: str) -> bool:
    """Check if line is a stage direction"""
    stripped = line.strip()
    
    # Parenthetical direction
    if stripped.startswith('(') and stripped.endswith(')'):
        return True
    
    # F/X or SFX markers
    if re.match(r'^(F/X|SFX|FX|GRAMS|MUSIC)[\s:]', stripped, re.IGNORECASE):
        return True
    
    # Scene heading
    if re.match(r'^(INT|EXT|INT/EXT|INTERIOR|EXTERIOR)[\.\s]', stripped, re.IGNORECASE):
        return True
    
    # Numbered scene
    if re.match(r'^\d+[:\.]?\s*(INT|EXT)', stripped, re.IGNORECASE):
        return True
    
    return False


def parse_part_content(lines: list[str], start: int, end: int) -> tuple[list[ScriptElement], str]:
    """
    Parse the content of a single part into structured elements.
    Returns (elements, searchable_dialogue_text)
    """
    elements = []
    dialogue_parts = []
    current_character = None
    current_dialogue = []
    
    i = start
    while i <= end:
        line = lines[i].strip()
        
        if not line:
            # Empty line - flush current dialogue
            if current_character and current_dialogue:
                dialogue_text = ' '.join(current_dialogue)
                elements.append(ScriptElement(
                    type=ElementType.DIALOGUE,
                    character=current_character,
                    text=dialogue_text
                ))
                dialogue_parts.append(f"{current_character}: {dialogue_text}")
                current_dialogue = []
            i += 1
            continue
        
        # Check for stage direction
        if is_stage_direction(line):
            # Flush current dialogue first
            if current_character and current_dialogue:
                dialogue_text = ' '.join(current_dialogue)
                elements.append(ScriptElement(
                    type=ElementType.DIALOGUE,
                    character=current_character,
                    text=dialogue_text
                ))
                dialogue_parts.append(f"{current_character}: {dialogue_text}")
                current_dialogue = []
            
            # Add direction
            direction_text = line.strip('()')
            elements.append(ScriptElement(
                type=ElementType.DIRECTION,
                text=direction_text
            ))
            i += 1
            continue
        
        # Check for character line
        char, same_line_dialogue = is_character_line(line)
        if char:
            # Flush previous character's dialogue
            if current_character and current_dialogue:
                dialogue_text = ' '.join(current_dialogue)
                elements.append(ScriptElement(
                    type=ElementType.DIALOGUE,
                    character=current_character,
                    text=dialogue_text
                ))
                dialogue_parts.append(f"{current_character}: {dialogue_text}")
                current_dialogue = []
            
            current_character = char
            
            # Check if dialogue is on same line
            if same_line_dialogue:
                current_dialogue.append(same_line_dialogue)
            
            i += 1
            continue
        
        # Regular line - add to current dialogue or treat as continuation
        if current_character:
            current_dialogue.append(line)
        else:
            # No current character - might be narrative or continuation
            elements.append(ScriptElement(
                type=ElementType.DIRECTION,
                text=line
            ))
        
        i += 1
    
    # Flush final dialogue
    if current_character and current_dialogue:
        dialogue_text = ' '.join(current_dialogue)
        elements.append(ScriptElement(
            type=ElementType.DIALOGUE,
            character=current_character,
            text=dialogue_text
        ))
        dialogue_parts.append(f"{current_character}: {dialogue_text}")
    
    searchable = '\n'.join(dialogue_parts)
    return elements, searchable


def extract_cast_list(lines: list[str], preamble_end: int) -> list[dict]:
    """Extract cast list if present"""
    cast = []
    cast_pattern = r'^([A-Z][A-Z\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
    
    # Look in first 50 lines after preamble
    for i in range(preamble_end, min(preamble_end + 50, len(lines))):
        line = lines[i].strip()
        match = re.match(cast_pattern, line)
        if match:
            character = match.group(1).strip()
            actor = match.group(2).strip()
            # Filter out non-cast items
            if character not in ['PART', 'SCENE', 'INT', 'EXT', 'EPISODE']:
                cast.append({'character': character, 'actor': actor})
    
    return cast


def detect_format_type(lines: list[str], preamble_end: int) -> str:
    """Detect which format type this script uses"""
    # Check first 20 lines after preamble
    sample = '\n'.join(lines[preamble_end:preamble_end + 30])
    
    if re.search(r'^\[Part', sample, re.MULTILINE):
        return 'A'  # [Part One] style
    elif re.search(r'^PART (ONE|TWO|1|2)', sample, re.MULTILINE):
        return 'B'  # PART ONE style
    elif re.search(r'^EPISODE \d', sample, re.MULTILINE):
        return 'D'  # EPISODE style
    elif re.search(r'^\d+:\s*(INT|EXT)', sample, re.MULTILINE):
        return 'D'  # Numbered scenes
    else:
        return 'B'  # Default


def extract_title_author_from_content(lines: list[str]) -> tuple[str, str]:
    """Extract title and author from script content"""
    title = ""
    author = ""
    
    # Look in first 20 lines
    for i, line in enumerate(lines[:20]):
        stripped = line.strip()
        
        # Title patterns
        if not title:
            # "Title, by Author" pattern
            match = re.match(r'^(.+?),?\s+by\s+(.+)$', stripped, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                author = match.group(2).strip()
                continue
            
            # "TITLE" alone (all caps, substantial)
            if re.match(r'^[A-Z][A-Z\s]{5,50}$', stripped):
                if not re.match(r'^(PART|SCENE|EPISODE|CAST|INT|EXT)', stripped):
                    title = stripped.title()
        
        # "By Author" pattern
        if not author:
            match = re.match(r'^By\s+(.+)$', stripped)
            if match:
                author = match.group(1).strip()
    
    return title, author


def parse_script(filepath: str) -> ParsedScript:
    """Parse a single script file into structured format"""
    filename = os.path.basename(filepath)
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Clean OCR errors
    content = clean_ocr_text(content)
    lines = content.split('\n')
    
    # Extract metadata
    release_num = extract_release_number(filename)
    title_from_file, author_from_file = extract_title_from_filename(filename)
    title_from_content, author_from_content = extract_title_author_from_content(lines)
    
    # Prefer filename metadata when it includes author (more reliable)
    # Content extraction can pick up preamble titles instead of actual script title
    if author_from_file:
        title = title_from_file
        author = author_from_file
    else:
        title = title_from_content or title_from_file
        author = author_from_content or ""
    
    # Find where script actually starts
    preamble_end = find_preamble_end(lines)
    
    # Detect format
    format_type = detect_format_type(lines, preamble_end)
    
    # Extract cast list
    cast_list = extract_cast_list(lines, preamble_end)
    
    # Find part boundaries
    part_boundaries = find_part_boundaries(lines, preamble_end)
    
    # Parse each part
    parts = []
    warnings = []
    
    for part_num, start, end in part_boundaries:
        elements, searchable = parse_part_content(lines, start, end)
        
        part = ScriptPart(
            part_number=part_num,
            start_line=start,
            end_line=end,
            elements=[asdict(e) for e in elements],
            searchable_dialogue=searchable
        )
        parts.append(asdict(part))
    
    # Generate ID
    script_id = f"{release_num:03d}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}"
    
    return ParsedScript(
        id=script_id,
        release_number=release_num,
        title=title,
        author=author,
        filename=filename,
        format_type=format_type,
        preamble_end_line=preamble_end,
        parts=parts,
        cast_list=cast_list,
        parse_warnings=warnings
    )


def parse_all_scripts(input_dir: str, output_dir: str) -> dict:
    """Parse all scripts in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'scripts': [],
        'errors': []
    }
    
    # Find all script files
    script_files = sorted(input_path.glob('*.txt'))
    results['total'] = len(script_files)
    
    for filepath in script_files:
        try:
            print(f"Parsing: {filepath.name}")
            script = parse_script(str(filepath))
            
            # Save individual script JSON
            script_dict = asdict(script)
            output_file = output_path / f"{script.id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(script_dict, f, indent=2, ensure_ascii=False)
            
            results['scripts'].append({
                'id': script.id,
                'release_number': script.release_number,
                'title': script.title,
                'author': script.author,
                'parts_count': len(script.parts),
                'format_type': script.format_type
            })
            results['successful'] += 1
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'file': filepath.name,
                'error': str(e)
            })
            print(f"  ERROR: {e}")
    
    # Save index
    index_file = output_path / '_index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script_parser.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Parsing scripts from: {input_dir}")
    print(f"Output to: {output_dir}")
    print()
    
    results = parse_all_scripts(input_dir, output_dir)
    
    print()
    print(f"=== RESULTS ===")
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print()
        print("Errors:")
        for err in results['errors']:
            print(f"  {err['file']}: {err['error']}")
