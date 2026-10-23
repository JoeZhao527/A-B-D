"""Turn an agent's raw message / reasoning text into NATURAL-LANGUAGE PROSE,
stripping the code / tool / terminal noise that would otherwise pollute a
dictionary counter (LIWC counts every token as a "word"). Table 7 of the paper
lists the rules.

The cleaning is deliberately conservative and auditable: every removal is a named
regex, and `build_corpus.py --show N` prints before/after samples for inspection.
"""
import json
import re

# ---- structural strippers (order matters) ----
_FENCED = re.compile(r"```.*?```", re.DOTALL)            # ``` code fences ```
_FENCED_OPEN = re.compile(r"```.*", re.DOTALL)           # unclosed trailing fence
_XML_THINK = re.compile(r"</?(mm:)?think>", re.I)        # MiniMax <mm:think> etc.
_HTML = re.compile(r"<[^>\n]{1,200}>")                   # inline tags
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PATH = re.compile(r"(?:\.{0,2}/)?(?:[\w.\-]+/){2,}[\w.\-]*")  # a/b/c paths
_LONGHEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")           # hashes / ids
_LONGNUM = re.compile(r"\b\d{7,}\b")                     # long digit runs
_TERM_ECHO = re.compile(r"^\s*(root@|\$ |# |>>> |\.\.\. )", re.M)  # prompt lines

# DCAgent-style content is a JSON object with analysis/plan/commands: keep the
# prose fields, drop the commands. (Small population, but handled cleanly.)
_JSON_PROSE_KEYS = ("analysis", "plan", "thought", "reasoning", "reflection")

# a "line" is noise if it is mostly non-alphabetic (code, tables, hexdumps).
_ALPHA = re.compile(r"[A-Za-z]")


def _maybe_json_prose(text: str):
    """If the whole message is a JSON dict (DCAgent format), return the
    concatenation of its prose fields; else None."""
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    parts = [str(obj[k]) for k in _JSON_PROSE_KEYS
             if k in obj and isinstance(obj[k], str) and obj[k].strip()]
    return "\n".join(parts) if parts else ""


def _line_is_prose(line: str, min_alpha_ratio: float = 0.55) -> bool:
    l = line.strip()
    if not l:
        return False
    letters = len(_ALPHA.findall(l))
    ratio = letters / max(1, len(l))
    # keep lines with enough letters and at least a couple of words
    return ratio >= min_alpha_ratio and len(l.split()) >= 2


def clean_prose(text) -> str:
    """Raw model text -> natural-language prose (best effort)."""
    if not text:
        return ""
    text = str(text)

    # 0) DCAgent JSON: reduce to its prose fields first
    jp = _maybe_json_prose(text)
    if jp is not None:
        text = jp

    # 1) strip structured spans
    text = _FENCED.sub(" ", text)
    text = _FENCED_OPEN.sub(" ", text)     # any dangling opened fence
    text = _XML_THINK.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PATH.sub(" ", text)
    text = _LONGHEX.sub(" ", text)
    text = _LONGNUM.sub(" ", text)

    # 2) drop terminal-echo / prompt lines and non-prose lines
    kept = [ln for ln in text.splitlines()
            if not _TERM_ECHO.match(ln) and _line_is_prose(ln)]
    text = " ".join(kept)

    # 3) strip leading markdown markers, collapse whitespace
    text = re.sub(r"(?m)^\s*[#>\-\*\d\.\)]+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split()) if text else 0
