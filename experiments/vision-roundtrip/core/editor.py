"""Text description editor for vision roundtrip.

Parses edit instructions and applies them to image descriptions.
Supports structured find/replace and free-form edit instructions.
"""

import re
from typing import Optional, Tuple, Dict


def parse_edit(edit_string: str) -> Optional[Tuple[str, str]]:
    """
    Parse an edit instruction into (find_text, replace_text).

    Supported formats:
      - change 'X' to 'Y'
      - replace 'X' with 'Y'
      - change "X" to "Y"
      - replace "X" with "Y"

    Returns None if the format is not recognised.
    """
    if not edit_string:
        return None

    # Try single-quoted: change 'X' to 'Y'
    match = re.match(
        r"(?:change|replace)\s+'([^']+)'\s+(?:to|with)\s+'([^']+)'",
        edit_string, re.IGNORECASE
    )
    if match:
        return match.group(1), match.group(2)

    # Try double-quoted: change "X" to "Y"
    match = re.match(
        r'(?:change|replace)\s+"([^"]+)"\s+(?:to|with)\s+"([^"]+)"',
        edit_string, re.IGNORECASE
    )
    if match:
        return match.group(1), match.group(2)

    return None


def apply_edit(original_text: str, edit_string: str) -> str:
    """
    Apply an edit instruction to a text description.

    If the edit can be parsed as find/replace, performs case-insensitive
    replacement of ALL occurrences.

    If the edit cannot be parsed, appends the edit instruction as a
    modification note (the LLM inversion will use it as guidance).
    """
    parsed = parse_edit(edit_string)

    if parsed:
        find_text, replace_text = parsed
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)
        if pattern.search(original_text):
            return pattern.sub(replace_text, original_text)
        else:
            # Find text not present — append as instruction
            return original_text + f"\n\nModification: {edit_string}"

    # Unparseable — append as generic modification instruction
    return original_text + f"\n\nModification: {edit_string}"


def apply_structured_edit(original_text: str, edits: Dict[str, str]) -> str:
    """
    Apply multiple find/replace edits to a text description.

    Edits are applied sequentially in dict order. Each edit is
    case-insensitive and replaces ALL occurrences.
    """
    result = original_text
    for find_text, replace_text in edits.items():
        pattern = re.compile(re.escape(find_text), re.IGNORECASE)
        result = pattern.sub(replace_text, result)
    return result
