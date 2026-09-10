#Actionableitems , decision , questions 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.rate_limiter import rate_limiter
import os
import re


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        timeout=60,
        max_retries=3
    )


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )


# ── NEW: single combined call replacing the old 3 separate functions ──────────

COMBINED_PROMPT = """You are an expert meeting analyst. From the meeting transcript, extract THREE things.
You MUST format your response using EXACTLY these three section headers, in this order,
so the response can be parsed programmatically. Do not add extra headers or commentary.

### ACTION ITEMS
For each item provide:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, else write 'Not specified')
Format as a numbered list. If none found, write exactly: No action items found.

### KEY DECISIONS
Format as a numbered list. If none found, write exactly: No key decisions found.

### OPEN QUESTIONS
List all unresolved questions or topics needing follow-up. Format as a numbered list.
If none found, write exactly: No open questions found.
"""


@rate_limiter.retry_with_backoff(max_retries=3, base_delay=5)
def extract_all(transcript: str) -> dict:
    """
    Single API call that extracts action items, key decisions, and open
    questions together. Replaces 3 separate calls with 1 — cuts quota usage
    for this stage by two-thirds.

    Returns:
        {
            "action_items": str,
            "key_decisions": str,
            "open_questions": str,
        }
    """
    chain = build_chain(COMBINED_PROMPT)
    rate_limiter.wait_if_needed()
    raw = chain.invoke(transcript)

    return _parse_combined_response(raw)


def _parse_combined_response(raw: str) -> dict:
    """
    Splits the model's single response into the three sections using the
    fixed headers we asked for. Falls back gracefully if the model
    formats slightly differently than requested.
    """
    sections = {
        "action_items": "No action items found.",
        "key_decisions": "No key decisions found.",
        "open_questions": "No open questions found.",
    }

    # Split on the headers we told the model to use
    pattern = r"###\s*ACTION ITEMS\s*(.*?)###\s*KEY DECISIONS\s*(.*?)###\s*OPEN QUESTIONS\s*(.*)"
    match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)

    if match:
        sections["action_items"] = match.group(1).strip() or sections["action_items"]
        sections["key_decisions"] = match.group(2).strip() or sections["key_decisions"]
        sections["open_questions"] = match.group(3).strip() or sections["open_questions"]
    else:
        # Fallback: model didn't follow the header format exactly.
        # Return the whole raw response under action_items so nothing is lost,
        # rather than silently dropping the output.
        sections["action_items"] = raw.strip()
        sections["key_decisions"] = "(Could not parse — see action items above for full response)"
        sections["open_questions"] = "(Could not parse — see action items above for full response)"

    return sections


# ── Old functions kept as thin wrappers so existing code calling them ─────────
# ── individually still works, but they now share one cached call ──────────────

_cache = {}


def _get_or_extract(transcript: str) -> dict:
    """Avoids re-calling the API if all three old functions are invoked
    on the same transcript in the same run."""
    key = hash(transcript)
    if key not in _cache:
        _cache[key] = extract_all(transcript)
    return _cache[key]


def extract_action_items(transcript: str) -> str:
    return _get_or_extract(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return _get_or_extract(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return _get_or_extract(transcript)["open_questions"]
