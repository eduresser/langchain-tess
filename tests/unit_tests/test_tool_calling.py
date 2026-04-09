"""Tests for JSON-only protocol tool calling, parsing, and retry logic."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_tessai import ChatTessAI, ToolCallParseError
from langchain_tessai.tool_calling import (
    IncrementalJsonContentExtractor,
    build_json_prompt,
    deep_parse_json,
    format_tools_for_prompt,
    parse_json_response,
    parse_json_string,
    validate_tool_call_contract,
)

API_KEY = "test-api-key-123"
AGENT_ID = 8794
BASE_URL = "https://api.tess.im"

SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name",
                }
            },
            "required": ["city"],
        },
    },
}


def _make_llm(**kwargs: Any) -> ChatTessAI:
    defaults = {
        "api_key": API_KEY,
        "agent_id": AGENT_ID,
        "workspace_id": 1,
        "model": "tess-5",
        "temperature": 0.5,
        "max_retries": 0,
    }
    defaults.update(kwargs)
    return ChatTessAI(**defaults)


def _make_execute_response(output: str, status: str = "succeeded") -> dict:
    return {
        "template_id": "8794",
        "responses": [
            {
                "id": 9001,
                "status": status,
                "input": "test",
                "output": output,
                "credits": 0.01,
                "root_id": 9001,
                "template_id": 8794,
            }
        ],
    }


def _mock_http_response(output: str, status: str = "succeeded") -> httpx.Response:
    return httpx.Response(
        200,
        json=_make_execute_response(output, status),
        request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
    )


def _json_content(text: str) -> str:
    return json.dumps({"content": text})


def _json_tool_call(content: str, name: str, arguments: dict) -> str:
    return json.dumps({
        "content": content,
        "commands": [{"name": name, "arguments": arguments}],
    })


# ==================================================================
# Tests: parse_json_string
# ==================================================================


class TestParseJsonString:
    def test_pure_json_object(self) -> None:
        result = parse_json_string('{"key": "value"}')
        assert result == {"key": "value"}

    def test_pure_json_array(self) -> None:
        result = parse_json_string('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_surrounded_by_text(self) -> None:
        result = parse_json_string('Here is the result: {"commands": []} end')
        assert result == {"commands": []}

    def test_no_json_returns_original(self) -> None:
        text = "This is just plain text without any JSON."
        result = parse_json_string(text)
        assert result == text

    def test_broken_json_returns_original(self) -> None:
        text = '{"key": broken_value}'
        result = parse_json_string(text)
        assert result == text

    def test_nested_braces(self) -> None:
        text = '{"outer": {"inner": "value"}}'
        result = parse_json_string(text)
        assert result == {"outer": {"inner": "value"}}

    def test_picks_first_balanced_json(self) -> None:
        text = 'prefix {"a": 1} middle {"b": 2} suffix'
        result = parse_json_string(text)
        assert result == {"a": 1}


# ==================================================================
# Tests: parse_json_response
# ==================================================================


class TestParseJsonResponse:
    def test_content_only(self) -> None:
        content, tc = parse_json_response('{"content": "Hello world"}')
        assert content == "Hello world"
        assert tc is None

    def test_content_with_tool_calls(self) -> None:
        raw = json.dumps({
            "content": "Searching...",
            "commands": [{"name": "get_weather", "arguments": {"city": "SP"}}],
        })
        content, tc = parse_json_response(raw)
        assert content == "Searching..."
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["name"] == "get_weather"
        assert tc[0]["args"] == {"city": "SP"}
        assert tc[0]["id"].startswith("call_")

    def test_empty_tool_calls_treated_as_none(self) -> None:
        raw = '{"content": "No tools needed", "commands": []}'
        content, tc = parse_json_response(raw)
        assert content == "No tools needed"
        assert tc is None

    def test_multiple_tool_calls(self) -> None:
        raw = json.dumps({
            "content": "",
            "commands": [
                {"name": "search", "arguments": {"q": "test"}},
                {"name": "get_weather", "arguments": {"city": "SP"}},
            ],
        })
        content, tc = parse_json_response(raw)
        assert content == ""
        assert tc is not None
        assert len(tc) == 2

    def test_truncates_text_after_json(self) -> None:
        raw = (
            '{"content": "Searching...", '
            '"commands": [{"name": "search", "arguments": {"q": "llm"}}]}'
            '\nNow let me also open... {"commands": [{"name": "open", "arguments": {}}]}'
            '\nHere are the hallucinated results...'
        )
        content, tc = parse_json_response(raw)
        assert content == "Searching..."
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["name"] == "search"

    def test_text_before_json_is_ignored(self) -> None:
        raw = 'Sure, here it is: {"content": "The answer", "commands": [{"name": "foo", "arguments": {"x": 1}}]}'
        content, tc = parse_json_response(raw)
        assert content == "The answer"
        assert tc is not None
        assert tc[0]["name"] == "foo"

    def test_not_json_raises(self) -> None:
        with pytest.raises(ToolCallParseError, match="not valid JSON"):
            parse_json_response("This is plain text with no JSON at all")

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ToolCallParseError, match="content"):
            parse_json_response('{"commands": [{"name": "x", "arguments": {}}]}')

    def test_content_not_string_raises(self) -> None:
        with pytest.raises(ToolCallParseError, match="content.*string"):
            parse_json_response('{"content": 42}')

    def test_invalid_tool_call_contract_raises(self) -> None:
        with pytest.raises(ToolCallParseError, match="arguments"):
            parse_json_response(
                '{"content": "", "commands": [{"name": "foo"}]}'
            )

    def test_tool_calls_before_content(self) -> None:
        raw = json.dumps({
            "commands": [{"name": "search", "arguments": {"q": "test"}}],
            "content": "Let me search",
        })
        content, tc = parse_json_response(raw)
        assert content == "Let me search"
        assert tc is not None
        assert tc[0]["name"] == "search"

    def test_stringified_arguments_are_deep_parsed(self) -> None:
        raw = json.dumps({
            "content": "",
            "commands": [{
                "name": "process",
                "arguments": {"config": '{"nested": true}'},
            }],
        })
        content, tc = parse_json_response(raw)
        assert tc is not None
        assert tc[0]["args"]["config"] == {"nested": True}

    def test_preserves_existing_id(self) -> None:
        raw = json.dumps({
            "content": "",
            "commands": [{"name": "foo", "arguments": {}, "id": "my-id"}],
        })
        _, tc = parse_json_response(raw)
        assert tc is not None
        assert tc[0]["id"] == "my-id"


# ==================================================================
# Tests: IncrementalJsonContentExtractor
# ==================================================================


class TestIncrementalExtractor:
    def test_simple_content_extraction(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"content": "Hello world"}'
        result = ext.feed(text)
        assert result == "Hello world"
        assert ext.content_complete

    def test_incremental_chunks(self) -> None:
        ext = IncrementalJsonContentExtractor()
        parts = ['{"con', 'tent": "He', 'llo ', 'world"}']
        collected = []
        for part in parts:
            collected.append(ext.feed(part))
        assert "".join(collected) == "Hello world"
        assert ext.content_complete

    def test_json_escape_handling(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"content": "line1\\nline2\\ttab\\"quote"}'
        result = ext.feed(text)
        assert result == 'line1\nline2\ttab"quote'

    def test_tool_calls_before_content_streams_content(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"commands": [{"name": "foo", "arguments": {}}], "content": "hi"}'
        result = ext.feed(text)
        assert result == "hi"
        assert ext.content_complete
        assert ext.get_full_text() == text

    def test_tool_calls_before_content_incremental(self) -> None:
        ext = IncrementalJsonContentExtractor()
        chunks = [
            '{"commands": [{"name": "search",',
            ' "arguments": {"q": "test"}}],',
            ' "content": "Let me ',
            'search for that"}',
        ]
        collected = []
        for ch in chunks:
            collected.append(ext.feed(ch))
        assert "".join(collected) == "Let me search for that"
        assert ext.content_complete

    def test_tool_calls_only_no_content(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=200)
        text = '{"commands": [{"name": "foo", "arguments": {}}]}'
        ext.feed(text)
        assert not ext.content_complete
        assert ext.get_full_text() == text

    def test_content_with_tool_calls_after(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"content": "Searching...", "commands": [{"name": "s", "arguments": {}}]}'
        result = ext.feed(text)
        assert result == "Searching..."
        assert ext.content_complete

    def test_get_full_text_returns_all_fed_chunks(self) -> None:
        ext = IncrementalJsonContentExtractor()
        ext.feed('{"content')
        ext.feed('": "hello')
        ext.feed('"}')
        assert ext.get_full_text() == '{"content": "hello"}'

    def test_unicode_escape(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"content": "caf\\u00e9"}'
        result = ext.feed(text)
        assert result == "café"

    def test_empty_content(self) -> None:
        ext = IncrementalJsonContentExtractor()
        text = '{"content": "", "commands": [{"name": "x", "arguments": {}}]}'
        result = ext.feed(text)
        assert result == ""
        assert ext.content_complete


class TestIncrementalExtractorPassthrough:
    """Tests for the passthrough fallback when the model responds with
    plain text instead of the JSON-only protocol."""

    def test_plain_text_triggers_passthrough(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=10)
        result = ext.feed("Hello world")
        assert result == "Hello world"
        assert ext.is_passthrough
        assert ext.content_complete

    def test_subsequent_chunks_forwarded_directly(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=10)
        collected = []
        collected.append(ext.feed("## Title --"))
        collected.append(ext.feed("\n\nSome text"))
        collected.append(ext.feed(" more text"))
        assert collected == ["## Title --", "\n\nSome text", " more text"]
        assert ext.is_passthrough

    def test_markdown_triggers_passthrough(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=20)
        result = ext.feed("# Attention in Transformers\n\n**Bold**")
        assert "# Attention" in result
        assert ext.is_passthrough

    def test_json_without_content_key_triggers_passthrough(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=30)
        collected = []
        collected.append(ext.feed('{"answer": "this is not the'))
        collected.append(ext.feed(' expected format"}'))
        full = "".join(collected)
        assert "answer" in full
        assert ext.is_passthrough

    def test_below_threshold_does_not_trigger(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=100)
        result = ext.feed("short text")
        assert result == ""
        assert not ext.is_passthrough
        assert not ext.content_complete

    def test_valid_json_does_not_trigger_passthrough(self) -> None:
        ext = IncrementalJsonContentExtractor()
        ext.feed('{"content": "hello"}')
        assert not ext.is_passthrough
        assert ext.content_complete

    def test_code_fenced_json_not_immediate_passthrough(self) -> None:
        """```json preamble should NOT trigger immediate passthrough;
        the threshold gives room for the real ``{`` to appear."""
        ext = IncrementalJsonContentExtractor(passthrough_threshold=50)
        ext.feed('```json\n{"content": "hello"}```')
        assert not ext.is_passthrough
        assert ext.content_complete

    def test_whitespace_before_json_works(self) -> None:
        ext = IncrementalJsonContentExtractor()
        result = ext.feed('  \n\t {"content": "hello"}')
        assert result == "hello"
        assert not ext.is_passthrough
        assert ext.content_complete

    def test_passthrough_get_full_text(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=10)
        ext.feed("chunk1-abcd")
        ext.feed("chunk2")
        assert ext.get_full_text() == "chunk1-abcdchunk2"
        assert ext.is_passthrough

    def test_passthrough_get_extracted_content_empty(self) -> None:
        """In passthrough mode the JSON-based content chars are empty
        (content was yielded directly via feed, not extracted from JSON)."""
        ext = IncrementalJsonContentExtractor(passthrough_threshold=5)
        ext.feed("Hello")
        assert ext.get_extracted_content() == ""
        assert ext.is_passthrough

    def test_threshold_not_reached_stays_seeking(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=100)
        result = ext.feed('{"cont')
        assert result == ""
        assert not ext.is_passthrough
        assert not ext.content_complete

    def test_threshold_reached_without_content_key(self) -> None:
        ext = IncrementalJsonContentExtractor(passthrough_threshold=20)
        collected = []
        collected.append(ext.feed('{"something_else": '))
        collected.append(ext.feed('"value", "more": 123'))
        full = "".join(collected)
        assert ext.is_passthrough
        assert "something_else" in full

    def test_incremental_plain_text_streaming(self) -> None:
        """Simulate realistic SSE streaming of plain text response."""
        ext = IncrementalJsonContentExtractor(passthrough_threshold=10)
        chunks = ["## O que é ", "Attention", "?\n\n", "É um mecanismo..."]
        collected = []
        for ch in chunks:
            collected.append(ext.feed(ch))
        assert collected[0] == "## O que é "
        assert all(collected[i] == chunks[i] for i in range(1, len(chunks)))
        assert ext.is_passthrough
        assert ext.content_complete


# ==================================================================
# Tests: deep_parse_json
# ==================================================================


class TestDeepParseJson:
    def test_string_with_json(self) -> None:
        result = deep_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_stringified_json(self) -> None:
        data = {"args": '{"city": "SP"}'}
        result = deep_parse_json(data)
        assert result == {"args": {"city": "SP"}}

    def test_list_with_stringified_json(self) -> None:
        data = ['{"a": 1}', "plain text"]
        result = deep_parse_json(data)
        assert result == [{"a": 1}, "plain text"]

    def test_plain_string_untouched(self) -> None:
        assert deep_parse_json("hello world") == "hello world"

    def test_non_string_passthrough(self) -> None:
        assert deep_parse_json(42) == 42
        assert deep_parse_json(True) is True
        assert deep_parse_json(None) is None


# ==================================================================
# Tests: format_tools_for_prompt
# ==================================================================


class TestFormatToolsForPrompt:
    def test_single_tool(self) -> None:
        result = format_tools_for_prompt([SAMPLE_TOOL])
        assert "get_weather" in result
        assert "city" in result
        assert "(required)" in result

    def test_multiple_tools(self) -> None:
        tool2 = {
            "type": "function",
            "function": {
                "name": "search_db",
                "description": "Search the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        }
        result = format_tools_for_prompt([SAMPLE_TOOL, tool2])
        assert "get_weather" in result
        assert "search_db" in result
        assert "(optional)" in result

    def test_tool_without_params(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result = format_tools_for_prompt([tool])
        assert "get_time" in result
        assert "(no parameters)" in result


# ==================================================================
# Tests: build_json_prompt
# ==================================================================


class TestBuildJsonPrompt:
    def test_prompt_with_tools(self) -> None:
        prompt = build_json_prompt([SAMPLE_TOOL])
        assert "You can perform the following commands" in prompt
        assert "commands" in prompt
        assert "get_weather" in prompt
        assert "single valid JSON object" in prompt

    def test_prompt_without_tools(self) -> None:
        prompt = build_json_prompt()
        assert "single valid JSON object" in prompt
        assert '"content"' in prompt
        assert "commands" not in prompt

    def test_prompt_contains_stop_rule(self) -> None:
        prompt = build_json_prompt([SAMPLE_TOOL])
        assert "Do NOT simulate" in prompt


# ==================================================================
# Tests: validate_tool_call_contract
# ==================================================================


class TestValidateContract:
    def test_valid_contract(self) -> None:
        data = {"commands": [{"name": "foo", "arguments": {"x": 1}}]}
        result = validate_tool_call_contract(data)
        assert len(result) == 1
        assert result[0]["name"] == "foo"
        assert result[0]["args"] == {"x": 1}

    def test_missing_tool_calls_key(self) -> None:
        with pytest.raises(ToolCallParseError, match="Missing"):
            validate_tool_call_contract({"other": "data"})

    def test_tool_calls_not_a_list(self) -> None:
        with pytest.raises(ToolCallParseError, match="must be a list"):
            validate_tool_call_contract({"commands": "not-a-list"})

    def test_empty_tool_calls(self) -> None:
        with pytest.raises(ToolCallParseError, match="empty"):
            validate_tool_call_contract({"commands": []})

    def test_missing_name(self) -> None:
        with pytest.raises(ToolCallParseError, match="name"):
            validate_tool_call_contract({"commands": [{"arguments": {}}]})

    def test_missing_arguments(self) -> None:
        with pytest.raises(ToolCallParseError, match="arguments"):
            validate_tool_call_contract({"commands": [{"name": "foo"}]})

    def test_arguments_not_dict(self) -> None:
        with pytest.raises(ToolCallParseError, match="arguments.*dict"):
            validate_tool_call_contract(
                {"commands": [{"name": "foo", "arguments": "bad"}]}
            )

    def test_item_not_dict(self) -> None:
        with pytest.raises(ToolCallParseError, match="must be a dict"):
            validate_tool_call_contract({"commands": ["not-a-dict"]})


# ==================================================================
# Tests: JSON format prompt injection
# ==================================================================


class TestJsonFormatPromptInjection:
    def test_injects_tool_prompt_as_system_message(self) -> None:
        msgs = [HumanMessage(content="hello")]
        result = ChatTessAI._inject_json_format_prompt(msgs, [SAMPLE_TOOL])
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert "get_weather" in result[0].content
        assert "JSON" in result[0].content

    def test_injects_json_prompt_without_tools(self) -> None:
        msgs = [HumanMessage(content="hello")]
        result = ChatTessAI._inject_json_format_prompt(msgs, None)
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert "JSON" in result[0].content
        assert "commands" not in result[0].content

    def test_merges_with_existing_system(self) -> None:
        msgs = [
            SystemMessage(content="Be helpful."),
            HumanMessage(content="hello"),
        ]
        result = ChatTessAI._inject_json_format_prompt(msgs, [SAMPLE_TOOL])
        assert len(result) == 2
        assert isinstance(result[0], SystemMessage)
        assert "get_weather" in result[0].content
        assert "Be helpful." in result[0].content


# ==================================================================
# Tests: bind_tools + invoke  (JSON protocol)
# ==================================================================


class TestBindToolsInvoke:
    def test_invoke_with_tool_call_response(self) -> None:
        llm = _make_llm()
        tool_response = _json_tool_call("Searching...", "get_weather", {"city": "SP"})
        mock_resp = _mock_http_response(tool_response)

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            result = llm_with_tools.invoke("What's the weather?")

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"city": "SP"}
        assert result.content == "Searching..."

    def test_invoke_with_content_only_response(self) -> None:
        llm = _make_llm()
        mock_resp = _mock_http_response(_json_content("I don't need any tools for this."))

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            result = llm_with_tools.invoke("Hello!")

        assert isinstance(result, AIMessage)
        assert result.content == "I don't need any tools for this."
        assert not result.tool_calls

    def test_invoke_without_tools_uses_json_protocol(self) -> None:
        llm = _make_llm()
        mock_resp = _mock_http_response(_json_content("Hello there!"))

        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = llm.invoke("Hi!")

        assert isinstance(result, AIMessage)
        assert result.content == "Hello there!"


# ==================================================================
# Tests: retry on empty response
# ==================================================================


class TestRetryEmptyResponse:
    def test_retries_on_empty_then_succeeds(self) -> None:
        llm = _make_llm(max_retries=2)
        empty_resp = _mock_http_response("")
        ok_resp = _mock_http_response(_json_content("Finally got a response!"))

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return empty_resp
            return ok_resp

        with patch.object(httpx.Client, "post", side_effect=mock_post), \
             patch("langchain_tessai.chat_models.time.sleep"):
            result = llm.invoke("hello")

        assert result.content == "Finally got a response!"
        assert call_count == 3

    def test_raises_after_all_retries_exhausted(self) -> None:
        llm = _make_llm(max_retries=1)
        empty_resp = _mock_http_response("")

        with patch.object(httpx.Client, "post", return_value=empty_resp), \
             patch("langchain_tessai.chat_models.time.sleep"):
            with pytest.raises(ValueError, match="empty response"):
                llm.invoke("hello")


# ==================================================================
# Tests: retry on HTTP error
# ==================================================================


class TestRetryHTTPError:
    def test_retries_on_http_error_then_succeeds(self) -> None:
        llm = _make_llm(max_retries=1)
        error_resp = httpx.Response(
            500,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        ok_resp = _mock_http_response(_json_content("Success after retry!"))

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return error_resp
            return ok_resp

        with patch.object(httpx.Client, "post", side_effect=mock_post), \
             patch("langchain_tessai.chat_models.time.sleep"):
            result = llm.invoke("hello")

        assert result.content == "Success after retry!"

    def test_raises_http_error_after_retries(self) -> None:
        from langchain_tessai.exceptions import TessAPIError

        llm = _make_llm(max_retries=1)
        error_resp = httpx.Response(
            500,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=error_resp), \
             patch("langchain_tessai.chat_models.time.sleep"):
            with pytest.raises((httpx.HTTPStatusError, TessAPIError)):
                llm.invoke("hello")


# ==================================================================
# Tests: retry on invalid JSON
# ==================================================================


class TestRetryInvalidJson:
    def test_retries_invalid_json_then_succeeds(self) -> None:
        llm = _make_llm(max_retries=2)
        bad_resp = _mock_http_response("This is not JSON at all!")
        good_resp = _mock_http_response(
            _json_tool_call("", "get_weather", {"city": "SP"})
        )

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return bad_resp
            return good_resp

        with patch.object(httpx.Client, "post", side_effect=mock_post), \
             patch("langchain_tessai.chat_models.time.sleep"):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            result = llm_with_tools.invoke("What's the weather?")

        assert result.tool_calls[0]["name"] == "get_weather"
        assert call_count == 3

    def test_retries_contract_violation_then_succeeds(self) -> None:
        llm = _make_llm(max_retries=1)
        bad_resp = _mock_http_response('{"content": "", "commands": "not-a-list"}')
        good_resp = _mock_http_response(
            _json_tool_call("", "get_weather", {"city": "SP"})
        )

        call_count = 0

        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_resp
            return good_resp

        with patch.object(httpx.Client, "post", side_effect=mock_post), \
             patch("langchain_tessai.chat_models.time.sleep"):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            result = llm_with_tools.invoke("weather?")

        assert result.tool_calls[0]["name"] == "get_weather"

    def test_raises_after_retries_exhausted(self) -> None:
        llm = _make_llm(max_retries=1)
        bad_resp = _mock_http_response("Not JSON at all")

        with patch.object(httpx.Client, "post", return_value=bad_resp), \
             patch("langchain_tessai.chat_models.time.sleep"):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            with pytest.raises(ValueError, match="Failed to parse tool calls"):
                llm_with_tools.invoke("weather?")


# ==================================================================
# Tests: async retry
# ==================================================================


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_agenerate_retries_empty_then_succeeds(self) -> None:
        llm = _make_llm(max_retries=1)
        empty_resp = _mock_http_response("")
        ok_resp = _mock_http_response(_json_content("Async success!"))

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return empty_resp
            return ok_resp

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post), \
             patch("asyncio.sleep", return_value=None):
            result = await llm.ainvoke("hello")

        assert result.content == "Async success!"

    @pytest.mark.asyncio
    async def test_agenerate_with_tool_calls(self) -> None:
        llm = _make_llm()
        tool_resp = _mock_http_response(
            _json_tool_call("", "get_weather", {"city": "SP"})
        )

        with patch.object(httpx.AsyncClient, "post", return_value=tool_resp):
            llm_with_tools = llm.bind_tools([SAMPLE_TOOL])
            result = await llm_with_tools.ainvoke("weather?")

        assert result.tool_calls[0]["name"] == "get_weather"


# ==================================================================
# Tests: max_retries default
# ==================================================================


class TestMaxRetriesDefault:
    def test_default_max_retries_is_5(self) -> None:
        llm = ChatTessAI(api_key="key", agent_id=1, workspace_id=1)
        assert llm.max_retries == 5

    def test_max_retries_configurable(self) -> None:
        llm = ChatTessAI(api_key="key", agent_id=1, workspace_id=1, max_retries=10)
        assert llm.max_retries == 10


# ==================================================================
# Tests: build_tool_choice_instruction
# ==================================================================


from langchain_tessai.tool_calling import build_tool_choice_instruction


class TestBuildToolChoiceInstruction:
    def test_none_returns_empty(self) -> None:
        assert build_tool_choice_instruction(None) == ""

    def test_auto_returns_empty(self) -> None:
        assert build_tool_choice_instruction("auto") == ""

    def test_none_string_disables_tools(self) -> None:
        result = build_tool_choice_instruction("none")
        assert "Do NOT execute any commands" in result

    def test_false_disables_tools(self) -> None:
        result = build_tool_choice_instruction(False)
        assert "Do NOT execute any commands" in result

    def test_required_forces_tool_call(self) -> None:
        result = build_tool_choice_instruction("required")
        assert "MUST execute at least one command" in result

    def test_any_forces_tool_call(self) -> None:
        result = build_tool_choice_instruction("any")
        assert "MUST execute at least one command" in result

    def test_true_forces_tool_call(self) -> None:
        result = build_tool_choice_instruction(True)
        assert "MUST execute at least one command" in result

    def test_dict_with_function_name(self) -> None:
        result = build_tool_choice_instruction(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert 'MUST execute the command "get_weather"' in result

    def test_string_matching_tool_name(self) -> None:
        result = build_tool_choice_instruction("get_weather", [SAMPLE_TOOL])
        assert 'MUST execute the command "get_weather"' in result

    def test_string_not_matching_any_tool(self) -> None:
        result = build_tool_choice_instruction("unknown_tool", [SAMPLE_TOOL])
        assert result == ""


class TestBuildJsonPromptWithToolChoice:
    def test_prompt_with_tool_choice_required(self) -> None:
        prompt = build_json_prompt([SAMPLE_TOOL], tool_choice="required")
        assert "MUST execute at least one command" in prompt
        assert "get_weather" in prompt

    def test_prompt_with_tool_choice_none(self) -> None:
        prompt = build_json_prompt([SAMPLE_TOOL], tool_choice="none")
        assert "Do NOT execute any commands" in prompt

    def test_prompt_with_tool_choice_auto(self) -> None:
        prompt_auto = build_json_prompt([SAMPLE_TOOL], tool_choice="auto")
        prompt_no_choice = build_json_prompt([SAMPLE_TOOL])
        assert prompt_auto == prompt_no_choice

    def test_prompt_without_tools_ignores_tool_choice(self) -> None:
        prompt = build_json_prompt(None, tool_choice="required")
        assert "MUST call" not in prompt
