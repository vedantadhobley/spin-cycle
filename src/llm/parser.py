"""JSON extraction from LLM responses.

LLMs often wrap JSON in markdown code blocks, <think> tags, or preamble text.
This module extracts clean JSON from raw LLM output.
"""

import json
import re
from typing import Any, Optional

from src.utils.logging import log, get_logger

MODULE = "llm.parser"
logger = get_logger()


class JSONExtractionError(Exception):
    """Raised when JSON cannot be extracted from LLM output."""
    
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


def strip_think_tags(raw: str) -> tuple[str, Optional[str]]:
    """Strip <think>...</think> tags from reasoning model output.
    
    Args:
        raw: Raw LLM output
        
    Returns:
        Tuple of (content_after_think, thinking_content)
        - If <think> tags found: returns content after </think>, and the thinking
        - If no tags: returns original raw, None
    """
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1)
        after = raw[think_match.end():].strip()
        return after, thinking
    return raw, None


def extract_json(raw: str) -> Any:
    """Extract JSON from LLM output, handling common wrapper formats.
    
    Handles:
    - Raw JSON: {"key": "value"}
    - Markdown blocks: ```json\n{"key": "value"}\n```
    - <think> tags: <think>...</think>{"key": "value"}
    - Preamble text: "Here is the result:\n{"key": "value"}"
    - Trailing text: {"key": "value"}\nLet me know if you need anything else.
    
    Args:
        raw: Raw LLM output string
        
    Returns:
        Parsed JSON (dict or list)
        
    Raises:
        JSONExtractionError: If no valid JSON can be extracted
    """
    original_raw = raw
    raw = raw.strip()
    
    # Pre-process: strip <think> tags
    raw, thinking = strip_think_tags(raw)
    if thinking:
        log.debug(logger, MODULE, "stripped_think", "Stripped <think> tags from response")
    
    # Try 1: Direct parse (ideal case)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try 2: Extract from markdown code block
    # Handles ```json\n...\n``` or just ```\n...\n```
    code_block_match = re.search(
        r'```(?:json)?\s*\n?(.*?)\n?```',
        raw,
        re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try 3: Find JSON object or array with greedy matching
    # Look for outermost { } or [ ]
    json_patterns = [
        # Object: find the LAST } that balances the FIRST {
        (r'\{', r'\}'),
        # Array: find the LAST ] that balances the FIRST [
        (r'\[', r'\]'),
    ]
    
    for open_pat, close_pat in json_patterns:
        open_match = re.search(open_pat, raw)
        if open_match:
            # Find the balancing close bracket
            start = open_match.start()
            candidate = _extract_balanced(raw[start:], open_pat[0], close_pat[0])
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    
    # Try 4: If <think> tags ate everything, search original response
    if thinking and not raw:
        log.debug(logger, MODULE, "searching_original",
                 "Empty after <think>, searching original for JSON")
        decoder = json.JSONDecoder()
        for i, char in enumerate(original_raw):
            if char == '{':
                try:
                    parsed_candidate, _ = decoder.raw_decode(original_raw[i:])
                    if isinstance(parsed_candidate, dict):
                        return parsed_candidate
                except json.JSONDecodeError:
                    continue
    
    # Try 5: Last resort - find anything that looks like JSON
    last_resort = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', raw)
    if last_resort:
        try:
            return json.loads(last_resort.group(1))
        except json.JSONDecodeError:
            pass
    
    raise JSONExtractionError(
        f"Could not extract valid JSON from LLM output ({len(raw)} chars)",
        raw_output=original_raw
    )


def _extract_balanced(text: str, open_char: str, close_char: str) -> Optional[str]:
    """Extract a balanced bracket expression from text.
    
    Args:
        text: Text starting with open_char
        open_char: Opening bracket ('{' or '[')
        close_char: Closing bracket ('}' or ']')
        
    Returns:
        The balanced expression including brackets, or None if unbalanced
    """
    if not text or text[0] != open_char:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[:i + 1]
    
    return None  # Unbalanced


def safe_extract_json(raw: str, fallback: Any = None) -> tuple[Any, Optional[str]]:
    """Extract JSON with fallback on failure.
    
    Args:
        raw: Raw LLM output
        fallback: Value to return if extraction fails
        
    Returns:
        Tuple of (parsed_json, error_message)
        - On success: (parsed_json, None)
        - On failure: (fallback, error_message)
    """
    try:
        return extract_json(raw), None
    except JSONExtractionError as e:
        log.warning(logger, MODULE, "extract_failed",
                   "Failed to extract JSON from LLM output",
                   error=str(e), raw_length=len(raw))
        return fallback, str(e)
