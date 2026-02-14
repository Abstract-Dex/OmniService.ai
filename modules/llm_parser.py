"""
LLM-based manual parser using Claude Haiku.

Takes the raw extracted JSON string (from extract.py) and sends it to
Claude Haiku with the UnifiedModel JSON schema. The LLM returns structured
JSON which is validated into a Pydantic UnifiedModel.

Pipeline position:
    extract -> vector_ingest -> llm_parser (this) -> graph_store
"""

from modules.scheme import UnifiedModel
from anthropic import Anthropic
from dotenv import load_dotenv
import json
import re

load_dotenv()

# Anthropic client — reads ANTHROPIC_API_KEY from env
client = Anthropic()


def extract_json_from_response(text: str) -> str:
    """
    Strip markdown code fences (```json ... ```) from the LLM response.

    Claude sometimes wraps its JSON output in markdown fences even when
    told not to. This ensures we get the raw JSON string regardless.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def build_prompts(raw_json: str):
    """
    Build the system and user prompts for the LLM.

    Args:
        raw_json: The JSON string from extract.py (pages + text + tables).

    Returns:
        (system_prompt, user_prompt) tuple ready for the Anthropic API.
    """
    manual_json = json.loads(raw_json)

    # Generate the target schema from the Pydantic model so the LLM
    # knows exactly what shape of JSON to return
    schema = UnifiedModel.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    system_prompt = """You are a technical documentation parser.
    Extract structured data from manuals.
    Return ONLY valid JSON matching the provided schema.
    If information is not found, use null for optional fields.
    Do NOT hallucinate part numbers - only include those explicitly stated."""

    user_prompt = f"""Extract the refrigeration model data from the following manual content.

    ## Target JSON Schema:
    {schema_str}

    ## Hierarchy Rules:
    - Level 0: Complete System Assembly (root)
    - Level 1: Major assemblies (Compressor, Motor, Controller)
    - Level 2: Sub-components within assemblies (Capacitor, Relay, Probe)
    - Level 3: Individual parts (rarely used)

    ## Following is the reference manual content in the form of JSON:
    {manual_json}

    Return the extracted data only as a JSON object matching the RefrigerationModel schema and not markdown.
    """

    return system_prompt, user_prompt


def parse_manual(raw_json: str) -> UnifiedModel | None:
    """
    Send extracted manual JSON to Claude Haiku and return a validated UnifiedModel.

    Args:
        raw_json: JSON string from extract.py.

    Returns:
        A UnifiedModel instance, or None if the LLM call or validation fails.
    """
    system_prompt, user_prompt = build_prompts(raw_json)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            system=system_prompt,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        print(f"Error parsing manual: {e}")
        return None
    try:
        # Extract the text content and strip any code fences

        raw_text = response.content[0].text
        json_str = extract_json_from_response(raw_text)
        return UnifiedModel.model_validate_json(json_str)
    except Exception as e:
        print(f"Error parsing manual: {e}")
        return None
