"""Text cleanup utilities using LanguageTool.

LanguageTool is a grammar/style checker that runs locally. It catches:
- Spelling errors (basic)
- Grammar errors ("priming efforts" → sounds wrong)
- Style issues (wordiness, passive voice, etc.)

The Java server (~200MB) is downloaded on first use and runs locally.
We lazy-load it so startup isn't blocked.

Why we need this:
Quantized LLMs sometimes produce valid-word substitutions that spell checkers
miss (e.g., "priming" instead of "primary"). LanguageTool catches these
because they create grammatically odd phrases, even if each word is valid.
"""

import language_tool_python
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_tool() -> language_tool_python.LanguageTool:
    """Lazy-load LanguageTool instance.
    
    The first call downloads the Java server (~200MB) and starts it.
    Subsequent calls return the cached instance.
    """
    return language_tool_python.LanguageTool('en-US')


def cleanup_text(text: str | None) -> str | None:
    """Run text through LanguageTool and apply corrections.
    
    Args:
        text: The text to clean up. Can be None.
        
    Returns:
        Corrected text, or original text if cleanup fails.
        Returns None if input was None.
    """
    if not text:
        return text
    
    try:
        tool = _get_tool()
        matches = tool.check(text)
        
        # Apply corrections (language_tool_python provides this utility)
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected
        
    except Exception:
        # If anything goes wrong (Java not installed, server fails, etc.),
        # just return the original text. We don't want cleanup failures
        # to break the verification pipeline.
        return text
