"""JSON-based command calling support for Tess AI.

Since the Tess API does not support native OpenAI-style tool calling,
this module provides prompt engineering utilities to enforce a JSON-only
response protocol and parse structured command responses.

The model is always instructed to respond with a single JSON object:
  {"content": "...", "commands": [...]}

The term "commands" is used in the prompt instead of "tools"/"tool_calls"
to reduce model hallucination (models trained on tool-calling patterns
tend to simulate tool results when they see "tools" in the prompt).
Internally, commands are mapped to LangChain's standard tool_calls
interface.

The closing ``}`` of the JSON object provides a natural stop boundary --
any hallucinated text after it is trivially discarded by extracting only
the first balanced JSON object from the raw output.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


JSON_REMINDER_WITH_TOOLS = (
    '\n\n[ONE JSON ONLY] {"content": "full response here in markdown format", "commands": [...]}'
    " — put your ENTIRE answer in content, no text outside the JSON."
)

JSON_REMINDER_NO_TOOLS = (
    '\n\n[ONE JSON ONLY] {"content": "full response here in markdown format"}'
    " — put your ENTIRE answer in content."
)


class ToolCallParseError(ValueError):
    """Raised when the model response cannot be parsed as valid JSON.

    Carries the raw ``output`` so callers (e.g. the retry loop) can
    inspect what the model actually returned.
    """

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


# ------------------------------------------------------------------
# System prompts  (100 % English, JSON-only protocol)
# ------------------------------------------------------------------

JSON_RESPONSE_SYSTEM_PROMPT = """\
You must ALWAYS respond with a single valid JSON object and nothing else.

Format:
{{"content": "your response text"}}

Rules:
- Your entire response MUST be exactly one JSON object.
- The "content" field is REQUIRED and must be a string containing your reply.
- Do NOT write any text, markdown, or explanation outside the JSON object.
- Do NOT wrap the JSON in code fences or backticks."""

JSON_TOOL_CALLING_SYSTEM_PROMPT = """\
You can perform the following commands:

{tool_definitions}

You must ALWAYS respond with a single valid JSON object and nothing else.

When you need to execute one or more commands:
{{"content": "optional short explanation", "commands": [{{"name": "command_name", "arguments": {{"arg1": "value1"}}}}]}}

When you do NOT need to execute any command (i.e. you already have the answer):
{{"content": "your FULL and COMPLETE response here — put ALL text inside this field, including reports, analyses, tables, markdown, etc."}}

Rules:
- Your entire response MUST be exactly one JSON object.
- The "content" field is REQUIRED and must be a string. It supports VERY long text — put your entire response inside it, no matter how long (reports, analyses, tables, etc.).
- The "commands" field is OPTIONAL. Include it ONLY when you need to execute one of the commands listed above.
- The ONLY valid command names are the ones listed above. Do NOT invent or fabricate commands such as "respond", "reply", "answer", "fetch", "browse", or any other name not listed above.
- When you have the final answer, write your COMPLETE response inside "content". Do NOT create a command to deliver your answer — put everything in "content".
- Each command must have "name" (exactly matching a command name above) and \
"arguments" (a JSON object matching the command's parameter schema).
- You may execute multiple commands at once by adding multiple entries to the "commands" array.
- Do NOT write any text, markdown, or explanation outside the JSON object.
- Do NOT wrap the JSON in code fences or backticks.
- Do NOT simulate or imagine command results. You will receive results in the next message.
- NEVER generate more than one JSON object."""


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    """Convert a list of OpenAI-format tool dicts into a human-readable
    description suitable for injection into a system prompt.

    Each tool dict is expected to follow the OpenAI schema::

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { ... JSON Schema ... }
            }
        }
    """
    parts: list[str] = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        description = func.get("description", "No description provided.")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        param_lines: list[str] = []
        for pname, pschema in properties.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req_marker = " (required)" if pname in required else " (optional)"
            param_lines.append(f"    - {pname} ({ptype}{req_marker}): {pdesc}")

        params_block = "\n".join(param_lines) if param_lines else "    (no parameters)"
        parts.append(
            f"Command: {name}\n"
            f"  Description: {description}\n"
            f"  Parameters:\n{params_block}"
        )

    return "\n\n".join(parts)


def build_tool_choice_instruction(
    tool_choice: Optional[Union[dict, str, bool]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Return an instruction string to append to the system prompt based on
    *tool_choice*.

    Implements OpenAI-style ``tool_choice`` semantics via prompt
    engineering, since the Tess API has no native support for it.

    Returns an empty string when no extra instruction is needed
    (``None`` or ``"auto"``).
    """
    if tool_choice is None or tool_choice == "auto":
        return ""

    if tool_choice == "none" or tool_choice is False:
        return (
            "\n\nIMPORTANT: Do NOT execute any commands. "
            "Respond with content only. Do NOT include the \"commands\" field."
        )

    if tool_choice in ("required", "any") or tool_choice is True:
        return (
            "\n\nIMPORTANT: You MUST execute at least one command. "
            "The \"commands\" array is REQUIRED in your response."
        )

    if isinstance(tool_choice, dict):
        name = (
            tool_choice.get("function", {}).get("name")
            or tool_choice.get("name", "")
        )
        if name:
            return (
                f"\n\nIMPORTANT: You MUST execute the command \"{name}\". "
                f"No other command is allowed."
            )

    if isinstance(tool_choice, str):
        known_names = set()
        for t in tools or []:
            func = t.get("function", t)
            known_names.add(func.get("name", ""))
        if tool_choice in known_names:
            return (
                f"\n\nIMPORTANT: You MUST execute the command \"{tool_choice}\". "
                f"No other command is allowed."
            )

    return ""


def build_json_prompt(
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[dict, str, bool]] = None,
) -> str:
    """Build the system prompt enforcing JSON-only responses.

    When *tools* is provided, includes tool definitions and the
    ``tool_calls`` field documentation.  When *tool_choice* is
    provided, appends an instruction constraining which tools to call.
    Otherwise returns the simpler content-only JSON prompt.
    """
    if tools:
        prompt = JSON_TOOL_CALLING_SYSTEM_PROMPT.format(
            tool_definitions=format_tools_for_prompt(tools)
        )
        prompt += build_tool_choice_instruction(tool_choice, tools)
        return prompt
    return JSON_RESPONSE_SYSTEM_PROMPT


# ------------------------------------------------------------------
# JSON extraction  (port of parseJsonString / deepParseJson)
# ------------------------------------------------------------------


def _find_balanced_end(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Return the index of the closing delimiter that balances *open_ch*
    at *start*, skipping over JSON strings.  Returns ``-1`` on failure."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _get_trailing_content(text: str) -> str:
    """Return any non-whitespace text after the first balanced JSON object."""
    first_brace = text.find("{")
    first_bracket = text.find("[")

    candidates: list[tuple[int, str, str]] = []
    if first_brace != -1:
        candidates.append((first_brace, "{", "}"))
    if first_bracket != -1:
        candidates.append((first_bracket, "[", "]"))

    if not candidates:
        return ""

    candidates.sort(key=lambda c: c[0])

    for start, open_ch, close_ch in candidates:
        end = _find_balanced_end(text, start, open_ch, close_ch)
        if end == -1:
            continue
        return text[end + 1 :].strip()

    return ""


def has_trailing_content(text: str) -> bool:
    """Return ``True`` when *text* contains non-whitespace characters after
    the first balanced JSON object/array.
    """
    return len(_get_trailing_content(text)) > 0


def parse_json_string(text: str) -> Union[dict, list, str]:
    """Extract and parse the **first balanced** JSON structure from *text*.

    Uses brace/bracket counting (respecting JSON strings) so that
    trailing text after the first complete object/array is ignored.

    Returns the parsed object/array on success, or the original
    *text* unchanged if no valid JSON can be found.
    """
    first_brace = text.find("{")
    first_bracket = text.find("[")

    candidates: list[tuple[int, str, str]] = []
    if first_brace != -1:
        candidates.append((first_brace, "{", "}"))
    if first_bracket != -1:
        candidates.append((first_bracket, "[", "]"))

    if not candidates:
        return text

    candidates.sort(key=lambda c: c[0])

    for start, open_ch, close_ch in candidates:
        end = _find_balanced_end(text, start, open_ch, close_ch)
        if end == -1:
            continue
        json_str = text[start : end + 1]
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue

    return text


def deep_parse_json(value: Any) -> Any:
    """Recursively parse stringified-JSON values inside *value*.

    - If *value* is a ``str``, tries to parse it as JSON; if the result
      is a dict or list, recurses into it.
    - If *value* is a ``list``, recurses into each element.
    - If *value* is a ``dict``, recurses into each value.
    - Otherwise returns *value* unchanged.
    """
    if isinstance(value, str):
        parsed = parse_json_string(value)
        if parsed is not value and isinstance(parsed, (dict, list)):
            return deep_parse_json(parsed)
        return parsed

    if isinstance(value, list):
        return [deep_parse_json(item) for item in value]

    if isinstance(value, dict):
        return {k: deep_parse_json(v) for k, v in value.items()}

    return value


# ------------------------------------------------------------------
# Contract validation
# ------------------------------------------------------------------


def validate_tool_call_contract(data: dict, raw_output: str = "") -> List[dict]:
    """Validate that *data* follows the expected command contract.

    Returns the normalised list of command dicts on success.
    Raises ``ToolCallParseError`` on any contract violation.
    """
    tool_calls = data.get("commands")
    if tool_calls is None:
        raise ToolCallParseError(
            'Missing "commands" key in parsed JSON', raw_output
        )
    if not isinstance(tool_calls, list):
        raise ToolCallParseError(
            f'"commands" must be a list, got {type(tool_calls).__name__}',
            raw_output,
        )
    if len(tool_calls) == 0:
        raise ToolCallParseError(
            '"commands" array is empty', raw_output
        )

    validated: list[dict] = []
    for i, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            raise ToolCallParseError(
                f"tool_calls[{i}] must be a dict, got {type(tc).__name__}",
                raw_output,
            )
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            raise ToolCallParseError(
                f'tool_calls[{i}] missing or invalid "name"', raw_output
            )
        arguments = tc.get("arguments")
        if not isinstance(arguments, dict):
            raise ToolCallParseError(
                f'tool_calls[{i}] "arguments" must be a dict, '
                f"got {type(arguments).__name__}",
                raw_output,
            )
        validated.append(
            {
                "name": name,
                "args": arguments,
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            }
        )
    return validated


# ------------------------------------------------------------------
# JSON response parsing
# ------------------------------------------------------------------


def parse_json_response(text: str) -> Tuple[str, Optional[List[dict]]]:
    """Parse a JSON-protocol model response.

    Extracts the first balanced JSON object from *text* (discarding any
    trailing hallucinated content), then reads the ``content`` and
    optional ``tool_calls`` fields.

    Returns:
        ``(content, tool_calls)`` -- *tool_calls* is ``None`` when the
        model did not request any tool invocations.

    Raises:
        ``ToolCallParseError`` when the response is not valid JSON or
        violates the expected contract.
    """
    result = parse_json_string(text)

    if isinstance(result, str):
        raise ToolCallParseError(
            "Model response is not valid JSON", text
        )

    if not isinstance(result, dict):
        raise ToolCallParseError(
            f"Expected JSON object, got {type(result).__name__}", text
        )

    result = deep_parse_json(result)

    content = result.get("content")
    if content is None:
        raise ToolCallParseError(
            'Missing required "content" field in JSON response', text
        )
    if isinstance(content, (dict, list)):
        # deep_parse_json recursively parsed a stringified-JSON content field.
        # Serialize back to string rather than raising — this preserves all
        # information when the model double-encodes its response.
        content = json.dumps(content, ensure_ascii=False)
    if not isinstance(content, str):
        raise ToolCallParseError(
            f'"content" must be a string, got {type(content).__name__}', text
        )

    raw_tool_calls = result.get("commands")

    if raw_tool_calls is None:
        return content, None

    if isinstance(raw_tool_calls, list) and len(raw_tool_calls) == 0:
        return content, None

    validated = validate_tool_call_contract(result, text)
    return content, validated


# ------------------------------------------------------------------
# Incremental JSON content extractor  (for SSE streaming)
# ------------------------------------------------------------------


class _ExtractorState(Enum):
    SEEKING_CONTENT = auto()
    IN_CONTENT_STRING = auto()
    AFTER_CONTENT = auto()
    TRAILING_PASSTHROUGH = auto()
    PASSTHROUGH = auto()


_PASSTHROUGH_CHAR_THRESHOLD = 40


class IncrementalJsonContentExtractor:
    """State machine that incrementally extracts the ``"content"`` field
    value from a JSON object being streamed token-by-token via SSE.

    Usage::

        ext = IncrementalJsonContentExtractor()
        for sse_chunk in stream:
            text_to_yield = ext.feed(sse_chunk)
            if text_to_yield:
                yield text_to_yield
        full_raw = ext.get_full_text()
    """

    def __init__(self, passthrough_threshold: int = _PASSTHROUGH_CHAR_THRESHOLD) -> None:
        self._buffer: list[str] = []
        self._state = _ExtractorState.SEEKING_CONTENT
        self._escape_next = False
        self._content_chars: list[str] = []
        self._content_yielded = 0
        self._scan_pos = 0
        self._passthrough_threshold = passthrough_threshold
        self._seen_tool_calls = False

    # ---- public API ------------------------------------------------

    def feed(self, chunk: str) -> str:
        """Append *chunk* to the internal buffer and return any new
        ``content`` characters ready to be yielded to the caller.

        When the accumulated buffer exceeds *passthrough_threshold*
        characters without finding a ``"content"`` JSON key (or the
        first non-whitespace character is not ``{``), the extractor
        switches to **passthrough mode** and returns raw chunks directly
        -- enabling token-by-token streaming even when the model
        responds with plain text instead of the JSON-only protocol.
        """
        self._buffer.append(chunk)

        if self._state == _ExtractorState.PASSTHROUGH:
            return chunk

        if self._state == _ExtractorState.TRAILING_PASSTHROUGH:
            buf = self._full_buffer()
            after = buf[self._json_end_pos + 1:].lstrip()
            new_chars = after[self._trailing_yielded:]
            self._trailing_yielded = len(after)
            return new_chars

        if self._state == _ExtractorState.AFTER_CONTENT:
            trailing = self._check_json_closed_and_trailing()
            if trailing is not None:
                return trailing
            return ""

        if self._state == _ExtractorState.SEEKING_CONTENT:
            self._try_find_content_start()
            if self._state == _ExtractorState.SEEKING_CONTENT:
                if self._should_switch_to_passthrough():
                    self._state = _ExtractorState.PASSTHROUGH
                    return self._full_buffer()

        if self._state == _ExtractorState.IN_CONTENT_STRING:
            return self._extract_content_chars()

        return ""

    @property
    def content_complete(self) -> bool:
        """True once all streamable content has been yielded.

        This covers the normal case (closing ``"`` of the JSON content
        string found), trailing passthrough (post-JSON text being
        streamed), and plain passthrough mode.
        """
        return self._state in (
            _ExtractorState.AFTER_CONTENT,
            _ExtractorState.TRAILING_PASSTHROUGH,
            _ExtractorState.PASSTHROUGH,
        )

    @property
    def is_passthrough(self) -> bool:
        """True when the extractor gave up on JSON and streams raw text."""
        return self._state in (
            _ExtractorState.PASSTHROUGH,
            _ExtractorState.TRAILING_PASSTHROUGH,
        )

    def get_full_text(self) -> str:
        """Return the complete raw buffer for final JSON parsing."""
        return "".join(self._buffer)

    def get_extracted_content(self) -> str:
        """Return all content characters extracted so far."""
        return "".join(self._content_chars)

    # ---- internals -------------------------------------------------

    _json_end_pos: int = -1
    _trailing_yielded: int = 0

    def _full_buffer(self) -> str:
        return "".join(self._buffer)

    def _check_json_closed_and_trailing(self) -> Optional[str]:
        """After ``"content"`` extraction, look for the root JSON ``}``
        closing brace.  Once found, any non-whitespace text after it is
        trailing content that should be streamed directly.

        Returns the trailing text to yield, or ``None`` if the root JSON
        hasn't closed yet.
        """
        buf = self._full_buffer()
        if self._json_end_pos < 0:
            end = _find_balanced_end(buf, 0, "{", "}")
            if end < 0:
                return None
            self._json_end_pos = end

        after = buf[self._json_end_pos + 1:]
        stripped = after.lstrip()
        if not stripped:
            return None

        self._state = _ExtractorState.TRAILING_PASSTHROUGH
        self._trailing_yielded = len(stripped)
        return stripped

    def _should_switch_to_passthrough(self) -> bool:
        """Decide whether to abandon JSON extraction and stream raw text.

        Switches after *passthrough_threshold* characters have
        accumulated without finding ``"content"`` (or ``"commands"``).
        This covers plain-text responses, markdown, code-fenced JSON
        (` ```json\\n{...}``` `), and other non-protocol output -- while
        giving enough room for leading whitespace or small preambles
        before a valid ``{`` appears.

        Never triggers when ``"commands"`` has already been seen,
        since that confirms the JSON-only protocol is in use and
        ``"content"`` is expected to follow.
        """
        if self._seen_tool_calls:
            return False
        return len(self._full_buffer()) >= self._passthrough_threshold

    def _try_find_content_start(self) -> None:
        buf = self._full_buffer()
        needle = '"content"'
        tc_needle = '"commands"'
        idx = buf.find(needle, self._scan_pos)
        tc_idx = buf.find(tc_needle, self._scan_pos)

        if tc_idx != -1 and (idx == -1 or tc_idx < idx):
            # "commands" appeared first -- skip past it and keep
            # searching for "content" which may follow later.
            self._seen_tool_calls = True
            self._scan_pos = tc_idx + len(tc_needle)
            idx = buf.find(needle, self._scan_pos)

        if idx == -1:
            self._scan_pos = max(0, len(buf) - len(needle))
            return

        colon_idx = buf.find(":", idx + len(needle))
        if colon_idx == -1:
            return
        quote_idx = buf.find('"', colon_idx + 1)
        if quote_idx == -1:
            return

        self._state = _ExtractorState.IN_CONTENT_STRING
        self._scan_pos = quote_idx + 1

    def _extract_content_chars(self) -> str:
        buf = self._full_buffer()
        new_chars: list[str] = []
        i = self._scan_pos
        while i < len(buf):
            ch = buf[i]
            if self._escape_next:
                if ch == "n":
                    new_chars.append("\n")
                elif ch == "t":
                    new_chars.append("\t")
                elif ch == "r":
                    new_chars.append("\r")
                elif ch == '"':
                    new_chars.append('"')
                elif ch == "\\":
                    new_chars.append("\\")
                elif ch == "/":
                    new_chars.append("/")
                elif ch == "b":
                    new_chars.append("\b")
                elif ch == "f":
                    new_chars.append("\f")
                elif ch == "u":
                    if i + 4 < len(buf):
                        hex_str = buf[i + 1 : i + 5]
                        try:
                            new_chars.append(chr(int(hex_str, 16)))
                        except ValueError:
                            new_chars.append("\\u" + hex_str)
                        i += 4
                    else:
                        # Hex digits have not arrived yet in this chunk.
                        # Rewind scan_pos to the backslash (i - 1) so the
                        # next feed() call reprocesses the full \uXXXX sequence.
                        self._escape_next = False
                        self._scan_pos = i - 1
                        self._content_chars.extend(new_chars)
                        self._content_yielded = len(self._content_chars)
                        return "".join(new_chars)
                else:
                    new_chars.append(ch)
                self._escape_next = False
                i += 1
                continue
            if ch == "\\":
                self._escape_next = True
                i += 1
                continue
            if ch == '"':
                self._state = _ExtractorState.AFTER_CONTENT
                self._scan_pos = i + 1
                self._content_chars.extend(new_chars)
                self._content_yielded = len(self._content_chars)
                return "".join(new_chars)
            new_chars.append(ch)
            i += 1

        self._scan_pos = i
        self._content_chars.extend(new_chars)
        result = "".join(new_chars)
        self._content_yielded = len(self._content_chars)
        return result
