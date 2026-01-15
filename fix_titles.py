#!/usr/bin/env python3
"""
Fix story titles based on the official Big Finish release list.
This script updates the search index with correct titles and removes duplicate files.
"""

import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "parsed_scripts"

# Master title mapping: release_number -> correct title
# Based on official Big Finish releases
TITLE_MAP = {
    1: "The Sirens of Time",
    2: "Phantasmagoria",
    3: "Whispers of Terror",
    4: "The Land of the Dead",
    5: "The Fearmonger",
    6: "The Marian Conspiracy",
    7: "The Genocide Machine",
    8: "Red Dawn",
    9: "The Spectre of Lanyon Moor",
    10: "Winter for the Adept",
    11: "The Apocalypse Element",
    12: "The Fires of Vulcan",
    13: "The Shadow of the Scourge",
    14: "The Holy Terror",
    15: "The Mutant Phase",
    16: "Storm Warning",
    17: "Sword of Orion",
    18: "The Stones of Venice",
    19: "Minuet in Hell",
    20: "Loups-Garoux",
    21: "Dust Breeding",
    22: "Bloodtide",
    23: "Project Twilight",
    24: "The Eye of the Scorpion",
    25: "Colditz",
    26: "Primeval",
    27: "The One Doctor",
    28: "Invaders from Mars",
    29: "The Chimes of Midnight",
    30: "Seasons of Fear",
    31: "Embrace the Darkness",
    32: "The Time of the Daleks",
    33: "Neverland",
    34: "Spare Parts",
    35: "...ish",
    36: "The Rapture",
    37: "The Sandman",
    38: "The Church and the Crown",
    39: "Bang-Bang-a-Boom!",
    40: "Jubilee",
    41: "Nekromanteia",
    42: "The Dark Flame",
    43: "Doctor Who and the Pirates",
    44: "Creatures of Beauty",
    45: "Project Lazarus",
    46: "Flip-Flop",
    47: "Omega",
    48: "Davros",
    49: "Master",
    50: "Zagreus",
    51: "The Wormery",
    52: "Scherzo",
    53: "The Creed of the Kromon",
    54: "The Natural History of Fear",
    55: "The Twilight Kingdom",
    56: "The Axis of Insanity",
    57: "Arrangements for War",
    58: "The Harvest",
    59: "The Roof of the World",
    60: "Medicinal Purposes",
    61: "Faith Stealer",
    62: "The Last",
    63: "Caerdroia",
    64: "The Next Life",
    65: "The Juggernauts",
    66: "The Game",
    67: "Dreamtime",
    68: "Catch-1782",
    69: "Three's a Crowd",
    70: "Unregenerate!",
    71: "The Council of Nicaea",
    72: "Terror Firma",
    73: "Thicker Than Water",
    74: "Live 34",
    75: "Scaredy Cat",
    76: "Singularity",
    77: "Other Lives",
    78: "Pier Pressure",
    79: "Night Thoughts",
    80: "Time Works",
    81: "The Kingmaker",
    82: "The Settling",
    83: "Something Inside",
    84: "The Nowhere Place",
    85: "Red",
    86: "The Reaping",
    87: "The Gathering",
    88: "Memory Lane",
    89: "No Man's Land",
    90: "The Year of the Pig",
    91: "Circular Time",
    92: "Nocturne",
    93: "Renaissance of the Daleks",
    94: "I.D. / Urgent Calls",
    95: "Exotron / Urban Myths",
    96: "Valhalla",
    97: "The Wishing Beast / The Vanity Box",
    98: "Frozen Time",
    99: "Son of the Dragon",
    100: "100",
    101: "Absolution",
    102: "The Mind's Eye / Mission of the Viyrans",
    103: "The Girl Who Never Was",
    104: "The Bride of Peladon",
    105: "The Condemned",
    106: "The Dark Husband",
    107: "The Haunting of Thomas Brewster",
    108: "Assassin in the Limelight",
    109: "The Death Collectors / Spider's Shadow",
    110: "The Boy That Time Forgot",
    111: "The Doomwood Curse",
    112: "Kingdom of Silver / Keepsake",
    113: "Time Reef / A Perfect World",
    114: "Brotherhood of the Daleks",
    115: "Forty-Five",
    116: "The Raincloud Man",
    117: "The Judgement of Isskar",
    118: "The Destroyer of Delights",
    119: "The Chaos Pool",
    120: "The Magic Mousetrap",
    121: "The Enemy of the Daleks",
    122: "The Angel of Scutari",
    123: "The Company of Friends",
    124: "Patient Zero",
    125: "Paper Cuts",
    126: "Blue Forgotten Planet",
    127: "Castle of Fear",
    128: "The Eternal Summer",
    129: "Plague of the Daleks",
    130: "A Thousand Tiny Wings",
    131: "Survival of the Fittest",
    132: "The Architects of History",
    133: "City of Spires",
    134: "The Wreck of the Titan",
    135: "Legend of the Cybermen",
    136: "Cobwebs",
    137: "The Whispering Forest",
    138: "The Cradle of the Snake",
    139: "Project: Destiny",
    140: "A Death in the Family",
    141: "Lurkers at Sunlight's Edge",
    142: "The Demons of Red Lodge and Other Stories",
    143: "The Crimes of Thomas Brewster",
    144: "The Feast of Axos",
    145: "Industrial Evolution",
    146: "Heroes of Sontar",
    147: "Kiss of Death",
    148: "Rat Trap",
    149: "Robophobia",
    150: "Recorded Time and Other Stories",
    151: "The Doomsday Quatrain",
    152: "House of Blue Fire",
    153: "The Silver Turk",
    154: "The Witch from the Well",
    155: "Army of Death",
    156: "The Curse of Davros",
    157: "The Fourth Wall",
    158: "Wirrn Isle",
    159: "The Emerald Tiger",
    160: "The Jupiter Conjunction",
    161: "The Butcher of Brisbane",
    162: "Protect and Survive",
    163: "Black and White",
    164: "Gods and Monsters",
    165: "The Burning Prince",
    166: "The Acheron Pulse",
    167: "The Shadow Heart",
    168: "1001 Nights",
    169: "The Wrong Doctors",
    170: "Spaceport Fear",
    171: "The Seeds of War",
    172: "Eldrad Must Die!",
    173: "The Lady of Mercia",
    174: "Prisoners of Fate",
    175: "Persuasion",
    176: "Starlight Robbery",
    177: "Daleks Among Us",
    178: "1963: Fanfare for the Common Men",
    179: "1963: The Space Race",
    180: "1963: The Assassination Games",
    181: "Afterlife",
    182: "Antidote to Oblivion",
    183: "The Brood of Erys",
    184: "Scavenger",
    185: "Moonflesh",
    186: "Tomb Ship",
    187: "Masquerade",
    188: "Breaking Bubbles and Other Stories",
    189: "Revenge of the Swarm",
    190: "Mask of Tragedy",
    191: "Signs and Wonders",
    192: "The Widow's Assassin",
    193: "Masters of Earth",
    194: "The Rani Elite",
    195: "Mistfall",
    196: "Equilibrium",
    197: "The Entropy Plague",
    198: "The Defectors",
    199: "Last of the Cybermen",
    200: "The Secret History",
    201: "We Are the Daleks",
    202: "The Warehouse",
    203: "Terror of the Sontarans",
    204: "Criss-Cross",
    205: "Planet of the Rani",
    206: "Shield of the Jotunn",
    207: "You Are the Doctor and Other Stories",
    208: "The Waters of Amsterdam",
    209: "Aquitaine",
    210: "The Peterloo Massacre",
    211: "And You Will Obey Me",
    212: "Vampire of the Mind",
    213: "The Two Masters",
    214: "A Life of Crime",
    215: "Fiesta of the Damned",
    216: "Maker of Demons",
    217: "The Memory Bank and Other Stories",
    218: "Order of the Daleks",
    219: "Absolute Power",
    220: "Quicksilver",
    221: "The Star Men",
    222: "The Contingency Club",
    223: "Zaltys",
    224: "Alien Heart / Dalek Soul",
    225: "Vortex Ice / Cortex Fire",
    226: "Shadow Planet / World Apart",
    227: "The High Price of Parking",
    228: "The Blood Furnace",
    229: "The Silurian Candidate",
    230: "Time in Office",
    231: "The Behemoth",
    232: "The Middle",
    233: "Static",
    234: "Kingdom of Lies",
    235: "Ghost Walk",
    236: "Serpent in the Silver Mask",
    237: "The Helliax Rift",
    238: "The Lure of the Nomad",
    239: "Iron Bright",
    240: "Hour of the Cybermen",
    241: "Red Planets",
    242: "The Dispossessed",
    243: "The Quantum Possibility Engine",
    244: "Warlock's Cross",
    245: "Muse of Fire",
    246: "The Hunting Ground",
    247: "Devil in the Mist",
    248: "Black Thursday / Power Game",
    249: "The Kamelion Empire",
    250: "The Monsters of Gokroth",
    251: "The Moons of Vulpana",
    252: "An Alien Werewolf in London",
    253: "Memories of a Tyrant",
    254: "Emissary of the Daleks",
    255: "Harry Houdini's War",
    256: "Tartarus",
    257: "Interstitial / Feast of Fear",
    258: "Warzone / Conversion",
    259: "Blood on Santa's Claw and Other Stories",
    260: "Dark Universe",
    261: "The Psychic Circus",
    262: "Subterfuge",
    263: "Cry of the Vultriss",
    264: "Scorched Earth",
}

# Special release titles (non-main-range)
SPECIAL_TITLES = {
    "10da1x1": "Technophobia",
    "10da1x2": "Time Reaver",
    "10da1x3": "Death and the Queen",
    "bs03x2": "The Green Eyed Monsters",
    "cc07x09": "The Scorchies",
    "cc10x01": "The Mouthless Dead",
    "cc10x02": "The Story of Extinction",
    "cc10x03": "The Integral",
    "cc10x04": "The Edge",
    "dw090a": "Return of the Daleks",
    "dw142a": "The Four Doctors",
    "dw155a": "The Five Companions",
    "dw168a": "Night of the Stormcrow",
    "dw181a": "Trial of the Valeyard",
    "fp1x1": "The Eleven-Day Empire",
    "fp1x2": "The Shadow Play",
    "fp1x3": "Sabbath Dei",
    "fp1x3a": "Sabbath's Tarot Game",
    "fp1x4": "In the Year of the Cat",
    "fp1x5": "Movers",
    "fp1x6": "A Labyrinth of Histories",
    "sp-1-05": "The Light at the End",
    "dalek-empire-1-2": "Dalek Empire",
}

# Files to delete (wrong titles/duplicates)
FILES_TO_DELETE = [
    # Duplicates with wrong titles
    "009-the-grand-lanyon",  # Keep 009-the-spectre-of-lanyon-moor
    "010-10da1x1",  # Wrong prefix - should be 000-
    "010-10da1x2",
    "010-10da1x3",
    "020-of-wildtracks-and-werewolves",  # Keep 020-loups-garoux
    "028-opening-doctor-who-theme-tune-arranged",  # Wrong title
    "030-charley-says",  # Keep 030-seasons-of-fear
    "035-sometimes-hyphenated-or-preceded",  # Wrong title for ...ish
    "062-grown-ever-more-menacing-and-which-i-fear-cannot-now-be-lifted",  # Keep proper title
    "064-lamp-light-yes-yes",  # Wrong title
    "077-designed",  # Keep 077-other-lives
    "093-renaissance-of-the-daleks-from-a-story",  # Keep correct version
    "100-100-bc",  # Keep 100
    "101-c-rizz-absolution-noun-the-remission-of-sins-granted",  # Keep absolution
    "115-false-gods",  # Wrong title
    "117-the-key-2-time-the-judgement-of-isskar",  # Keep simpler version
    "119-the-key-2-time-the-chaos-pool",  # Keep simpler version
    # Doctor Who prefix duplicates - keep the cleaner versions
    "120-doctor-who-the-magic-mousetrap",
    "121-doctor-who-enemy-of-the-daleks",
    "122-doctor-who-the-angel-of-scutari",
    "123-doctor-who",
    "124-doctor-who",
    "125-doctor-who",
    "126-doctor-who",
    "127-a-four-part-story",
    "128-the-eternal",
    "129-dramatis-personae",
    "130-doctor-who",
    "131-episode-one",
    "132-docior-who",
    "133-doctor-who-city-of-spires",
    "134-a-four-part-story",
    "136-cobwebs-fifth-doctor-trilogy-1",
    "137-purity",
    "138-the-ckadle-of-the-snake",
    "139-doctor-who-project-destiny",
    "140-doctor-who-a-death-in-the-family",
    "141-doctor-who-lurkers-at-sunlight-s-edge",
    "142-doctor-who-the-demons-of-red-lodge",
    "143-doctor-who-the-gathering-swarm",
    "144-doctor-who-the-feast-of-axos",
    "145-doctor-who-industrial-evolution",
    "146-doctor-who-heroes-of-sontar",
    "147-doctor-who-kiss-of-death",
    "148-doctor-who-rat-trap",
    "149-doctor-who",
    "150-doctor-who-recorded-time",
    "152-doctor-who-house-of-blue-fire",
    "154-an-eighth-doctor-and-mary-shelley-adventure",
    "155-army-of-dearth",
    "156-doctar",
    "158-irodigal-wririn",
    "161-doctor-who-the-butcher-of-brisbane",
    "162-doctor-who-protect-and-survive",
    "164-doctor-who-gods-and-monsters",
    "165-doctor-who-the-burning-prince",
    "166-doctor-who-the-acheron-pulse",
    "167-doctor-who-the-shadow-heart",
    "169-doctor-who",
    "170-doctor-who-spaceport-fear",
    "171-doctor-who",
    "172-doctor-who-eldrad-must-die",
    "173-doctor-who-the-lady-of-mercia",
    "174-doctor-who-prisoners-of-fate",
    "175-fersursicn",
    "177-doctor-who-daleks-among-us",
    "179-doctor-who-the-space-race",
    "180-doctor-who-the-assassination-games",
    "185-doctor-who-moonflesh",
    "186-doctor-who-tomb-ship",
    "187-doctor-who-masquerade",
    "188-breaking-bubbles",
    "197-doctor-who-the-entropy-plague",
    "198-doctor",
    "206-doctor-who-snowblind",
    "207-y-a-t-d-1-you-are-the-doctor",
    "212-212-vampire-of-the-mind",
    "213-doctor-who-the-two-masters",
    "214-alife-of-crime",
    "217-217a-the-memory-bank",
    "219-219-absolute-power",
    "221-221-the-star-men",
    "222-222-the-contingency-club",
    "223-223-zaltys",
    "224-224a-alien-heart",
    "225-225a-vortex-ice",
    "226-226a-shadow-planet",
    "230-230-time-in-office",
    "236-donc-iloik",  # Wrong OCR title
    "244-by-steve-lyons",  # Wrong - author instead of title
    "248-the-incredible-power-game",  # Keep black-thursday
    "250-bi-bic",  # Wrong OCR
    "251-bi-bic",  # Wrong OCR
    "252-bi-bic",  # Wrong OCR
    "255-a-four-part-story",  # Wrong
    "256-the-doctor-peter-davison",  # Wrong
    "257-iterstitial",  # Missing subtitle
    "258-the-doctor",  # Wrong
    "259-and-other-stories",  # Wrong
    "261-bi-bic",  # Wrong OCR
]


def load_search_index():
    """Load the search index JSON."""
    path = SCRIPTS_DIR / "search_index.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_search_index(index):
    """Save the search index JSON."""
    path = SCRIPTS_DIR / "search_index.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def get_correct_title(script_id, release_number, current_title):
    """Get the correct title for a script."""
    # Check main range
    if release_number > 0 and release_number in TITLE_MAP:
        return TITLE_MAP[release_number]

    # Check special releases
    # Extract the code part (e.g., "10da1x1" from "000-10da1x1" or "010-10da1x1")
    parts = script_id.split('-', 1)
    if len(parts) > 1:
        code = parts[1].lower()
        if code in SPECIAL_TITLES:
            return SPECIAL_TITLES[code]

    # Return current title if no mapping found
    return current_title


def fix_titles():
    """Update titles in the search index."""
    print("Loading search index...")
    index = load_search_index()
    if not index:
        print("Error: Could not load search index")
        return False

    print(f"Found {len(index['scripts'])} scripts")
    print()

    # Track changes
    changed = 0

    for script_id, script_info in index['scripts'].items():
        release_num = script_info['release_number']
        old_title = script_info['title']
        new_title = get_correct_title(script_id, release_num, old_title)

        if new_title != old_title:
            print(f"  {script_id}: '{old_title}' -> '{new_title}'")
            script_info['title'] = new_title
            changed += 1

    if changed > 0:
        print()
        print(f"Updated {changed} titles")
        save_search_index(index)
        print("Saved search index")
    else:
        print("No title changes needed")

    return True


def delete_duplicate_files():
    """Delete duplicate/incorrect files."""
    print()
    print("Checking for duplicate files to delete...")

    deleted = 0
    for file_id in FILES_TO_DELETE:
        for ext in ['.json', '.enc']:
            filepath = SCRIPTS_DIR / f"{file_id}{ext}"
            if filepath.exists():
                print(f"  Deleting: {filepath.name}")
                filepath.unlink()
                deleted += 1

    print(f"Deleted {deleted} files")


def remove_deleted_from_index():
    """Remove deleted files from the search index."""
    print()
    print("Removing deleted entries from search index...")

    index = load_search_index()
    if not index:
        return

    to_remove = []
    for script_id in index['scripts'].keys():
        if script_id in FILES_TO_DELETE:
            to_remove.append(script_id)

    for script_id in to_remove:
        print(f"  Removing: {script_id}")
        del index['scripts'][script_id]

    if to_remove:
        save_search_index(index)
        print(f"Removed {len(to_remove)} entries from index")


def main():
    print("=" * 60)
    print("Big Finish Title Fixer")
    print("=" * 60)
    print()

    # Step 1: Fix titles in search index
    print("Step 1: Fixing titles in search index...")
    fix_titles()

    # Step 2: Delete duplicate files
    print()
    print("Step 2: Deleting duplicate/incorrect files...")
    delete_duplicate_files()

    # Step 3: Remove deleted entries from index
    print()
    print("Step 3: Cleaning up search index...")
    remove_deleted_from_index()

    print()
    print("=" * 60)
    print("Done! Remember to re-encrypt with: python script_crypto.py encrypt")
    print("=" * 60)


if __name__ == "__main__":
    main()
