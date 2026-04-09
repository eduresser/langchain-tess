"""Tests for ChatTessAI (JSON-only protocol)."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langchain_tessai import ChatTessAI
from langchain_tessai.chat_models import FileRef

API_KEY = "test-api-key-123"
AGENT_ID = 8794
BASE_URL = "https://api.tess.im"


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


def _jc(text: str) -> str:
    """Wrap text in the JSON-only content envelope."""
    return json.dumps({"content": text})


# ------------------------------------------------------------------
# Message conversion
# ------------------------------------------------------------------


class TestConvertMessages:
    def test_human_message(self) -> None:
        msgs = [HumanMessage(content="hello")]
        result = ChatTessAI._convert_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_ai_message_serialized_as_json(self) -> None:
        msgs = [AIMessage(content="hi there")]
        result = ChatTessAI._convert_messages(msgs)
        assert result == [
            {"role": "assistant", "content": '{"content": "hi there"}'}
        ]

    def test_ai_message_with_tool_calls(self) -> None:
        msgs = [AIMessage(
            content="Searching",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "c1"}],
        )]
        result = ChatTessAI._convert_messages(msgs)
        parsed = json.loads(result[0]["content"])
        assert parsed["content"] == "Searching"
        assert parsed["commands"][0]["name"] == "search"

    def test_system_message_becomes_developer(self) -> None:
        msgs = [SystemMessage(content="you are helpful")]
        result = ChatTessAI._convert_messages(msgs)
        assert result == [{"role": "developer", "content": "you are helpful"}]

    def test_full_conversation(self) -> None:
        msgs = [
            SystemMessage(content="be brief"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            HumanMessage(content="bye"),
        ]
        result = ChatTessAI._convert_messages(msgs)
        assert result[0] == {"role": "developer", "content": "be brief"}
        assert result[1] == {"role": "user", "content": "hello"}
        assert result[2] == {"role": "assistant", "content": '{"content": "hi"}'}
        assert result[3] == {"role": "user", "content": "bye"}


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


class TestProperties:
    def test_llm_type(self) -> None:
        llm = _make_llm()
        assert llm._llm_type == "tess-ai"

    def test_identifying_params(self) -> None:
        llm = _make_llm()
        params = llm._identifying_params
        assert params["model_name"] == "tess-5"
        assert params["agent_id"] == AGENT_ID
        assert params["temperature"] == 0.5
        assert params["tools"] == "no-tools"

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"TESSAI_API_KEY": "env-key-999"}):
            llm = ChatTessAI(agent_id=1, workspace_id=1)
            assert llm.api_key.get_secret_value() == "env-key-999"

    def test_headers_include_auth_and_workspace(self) -> None:
        llm = _make_llm(workspace_id=42)
        headers = llm._headers
        assert headers["Authorization"] == f"Bearer {API_KEY}"
        assert headers["x-workspace-id"] == "42"


# ------------------------------------------------------------------
# Payload building
# ------------------------------------------------------------------


class TestBuildPayload:
    def test_basic_payload(self) -> None:
        llm = _make_llm()
        msgs = [HumanMessage(content="hi")]
        payload = llm._build_payload(msgs, stream=False)
        assert payload["model"] == "tess-5"
        assert payload["temperature"] == "0.5"
        assert payload["tools"] == "no-tools"
        assert payload["wait_execution"] is True
        assert "stream" not in payload

    def test_streaming_payload(self) -> None:
        llm = _make_llm()
        msgs = [HumanMessage(content="hi")]
        payload = llm._build_payload(msgs, stream=True)
        assert payload["stream"] is True
        assert payload["wait_execution"] is False

    def test_file_ids_included(self) -> None:
        llm = _make_llm(file_ids=[10, 20])
        msgs = [HumanMessage(content="hi")]
        payload = llm._build_payload(msgs, stream=False)
        assert payload["file_ids"] == [10, 20]


# ------------------------------------------------------------------
# _generate (sync, wait_execution=True)
# ------------------------------------------------------------------


EXECUTE_RESPONSE_COMPLETED = {
    "template_id": "8794",
    "responses": [
        {
            "id": 5001,
            "status": "succeeded",
            "input": "hello",
            "output": '{"content": "Hello! How can I help you?"}',
            "credits": 0.006,
            "root_id": 5001,
            "created_at": "2025-01-05T19:35:21.000000Z",
            "updated_at": "2025-01-05T19:35:23.000000Z",
            "template_id": 8794,
        }
    ],
}

EXECUTE_RESPONSE_STARTING = {
    "template_id": "8794",
    "responses": [
        {
            "id": 5002,
            "status": "starting",
            "input": "hello",
            "output": "",
            "credits": 0,
            "root_id": 5002,
            "created_at": "2025-01-05T19:35:21.000000Z",
            "updated_at": "2025-01-05T19:35:21.000000Z",
            "template_id": 8794,
        }
    ],
}

POLL_RESPONSE_SUCCEEDED = {
    "id": 5002,
    "status": "succeeded",
    "input": "hello",
    "output": '{"content": "Polled result!"}',
    "credits": 0.01,
    "root_id": 5002,
    "created_at": "2025-01-05T19:35:21.000000Z",
    "updated_at": "2025-01-05T19:35:23.000000Z",
    "template_id": 8794,
}


class TestGenerate:
    def test_generate_with_wait_execution(self) -> None:
        llm = _make_llm(wait_execution=True)

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            result = llm.invoke("hello")

        assert isinstance(result, AIMessage)
        assert result.content == "Hello! How can I help you?"
        assert result.response_metadata["tess_response_id"] == 5001
        assert result.response_metadata["credits"] == 0.006

    def test_wait_execution_true_falls_back_to_polling(self) -> None:
        llm = _make_llm(wait_execution=True, polling_interval=0.01)

        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_STARTING,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        mock_poll = httpx.Response(
            200,
            json=POLL_RESPONSE_SUCCEEDED,
            request=httpx.Request("GET", f"{BASE_URL}/agent-responses/5002"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_execute), \
             patch.object(httpx.Client, "get", return_value=mock_poll):
            result = llm.invoke("hello")

        assert isinstance(result, AIMessage)
        assert result.content == "Polled result!"
        assert result.response_metadata["tess_response_id"] == 5002
        assert result.response_metadata["tess_root_id"] == 5002

    def test_generate_with_polling(self) -> None:
        llm = _make_llm(wait_execution=False, polling_interval=0.01)

        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_STARTING,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        mock_poll = httpx.Response(
            200,
            json=POLL_RESPONSE_SUCCEEDED,
            request=httpx.Request("GET", f"{BASE_URL}/agent-responses/5002"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_execute), \
             patch.object(httpx.Client, "get", return_value=mock_poll):
            result = llm.invoke("hello")

        assert isinstance(result, AIMessage)
        assert result.content == "Polled result!"

    def test_generate_failed_raises(self) -> None:
        llm = _make_llm(wait_execution=False, polling_interval=0.01)

        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_STARTING,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        failed_poll = httpx.Response(
            200,
            json={
                "id": 5002,
                "status": "failed",
                "output": "",
                "error": "Something went wrong",
                "credits": 0,
            },
            request=httpx.Request("GET", f"{BASE_URL}/agent-responses/5002"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_execute), \
             patch.object(httpx.Client, "get", return_value=failed_poll):
            with pytest.raises(RuntimeError, match="Something went wrong"):
                llm.invoke("hello")


# ------------------------------------------------------------------
# _stream (sync)
# ------------------------------------------------------------------


class TestStream:
    def test_stream_chunks(self) -> None:
        llm = _make_llm()
        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [
                    {
                        "id": 100,
                        "status": "succeeded",
                        "output": _jc("Hello world"),
                        "credits": 0.005,
                        "root_id": 100,
                        "template_id": 10,
                    }
                ],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        with patch.object(httpx.Client, "post", return_value=mock_response):
            chunks = list(llm.stream("hello"))
        assert any(c.content == "Hello world" for c in chunks)
        assert any(
            c.response_metadata.get("credits") == 0.005 for c in chunks
        )


# ------------------------------------------------------------------
# _agenerate (async)
# ------------------------------------------------------------------


class TestAsyncGenerate:
    @pytest.mark.asyncio
    async def test_agenerate(self) -> None:
        llm = _make_llm(wait_execution=True)

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            result = await llm.ainvoke("hello")

        assert isinstance(result, AIMessage)
        assert result.content == "Hello! How can I help you?"


# ------------------------------------------------------------------
# _astream (async) with IncrementalJsonContentExtractor
# ------------------------------------------------------------------


def _sse_lines(events: list[dict]) -> list[str]:
    lines = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    return lines


class TestAsyncStream:
    @pytest.mark.asyncio
    async def test_astream_extracts_content_from_json(self) -> None:
        llm = _make_llm()

        json_output = '{"content": "Async hello"}'
        events = [
            {"id": 200, "status": "running", "output": json_output, "error": None,
             "credits": None, "root_id": 200, "template_id": 10},
            {"id": 200, "status": "completed", "output": "", "error": None,
             "credits": 0.003, "root_id": 200, "template_id": 10},
        ]

        class FakeAsyncStreamResponse:
            def __init__(self) -> None:
                self.status_code = 200

            def raise_for_status(self) -> None:
                pass

            async def aiter_lines(self):
                for line in _sse_lines(events):
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeAsyncClient:
            def stream(self, *args, **kwargs):
                return FakeAsyncStreamResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch.object(llm, "_async_client", return_value=FakeAsyncClient()):
            chunks = []
            async for chunk in llm.astream("hello"):
                chunks.append(chunk)

        contents = [c.content for c in chunks if c.content]
        assert "Async hello" in "".join(contents)
        meta_chunks = [c for c in chunks if c.response_metadata.get("credits")]
        assert any(c.response_metadata["credits"] == 0.003 for c in meta_chunks)


# ------------------------------------------------------------------
# SSE parsing
# ------------------------------------------------------------------


class TestSSEParsing:
    def test_parse_sse_line_valid(self) -> None:
        line = 'data: {"id": 1, "status": "running", "output": "hi"}'
        result = ChatTessAI._parse_sse_line(line)
        assert result == {"id": 1, "status": "running", "output": "hi"}

    def test_parse_sse_line_done(self) -> None:
        result = ChatTessAI._parse_sse_line("data: [DONE]")
        assert result is None

    def test_parse_sse_line_empty(self) -> None:
        assert ChatTessAI._parse_sse_line("") is None
        assert ChatTessAI._parse_sse_line("   ") is None

    def test_parse_sse_line_non_data(self) -> None:
        assert ChatTessAI._parse_sse_line("event: message") is None

    def test_parse_sse_line_invalid_json(self) -> None:
        assert ChatTessAI._parse_sse_line("data: not-json") is None


# ------------------------------------------------------------------
# Response metadata: root_id extraction
# ------------------------------------------------------------------


class TestRootIdExtraction:
    def test_extract_output_and_metadata_includes_root_id(self) -> None:
        llm = _make_llm()
        data = {
            "responses": [
                {
                    "id": 100,
                    "root_id": 100,
                    "status": "succeeded",
                    "output": "ok",
                    "credits": 0.01,
                    "template_id": 10,
                }
            ]
        }
        _, meta = llm._extract_output_and_metadata(data)
        assert meta["tess_root_id"] == 100

    def test_extract_output_and_metadata_from_poll_includes_root_id(self) -> None:
        llm = _make_llm()
        data = {
            "id": 200,
            "root_id": 200,
            "status": "succeeded",
            "output": "polled",
            "credits": 0.01,
            "template_id": 10,
        }
        _, meta = llm._extract_output_and_metadata_from_poll(data)
        assert meta["tess_root_id"] == 200

    def test_generate_response_contains_root_id(self) -> None:
        llm = _make_llm(wait_execution=True)

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            result = llm.invoke("hello")

        assert result.response_metadata["tess_root_id"] == 5001


# ------------------------------------------------------------------
# _build_payload_from_converted
# ------------------------------------------------------------------


class TestBuildPayloadFromConverted:
    def test_payload_without_root_id(self) -> None:
        llm = _make_llm()
        msgs = [{"role": "user", "content": "hi"}]
        payload = llm._build_payload_from_converted(msgs, stream=False)
        assert "root_id" not in payload
        assert payload["messages"] == msgs

    def test_payload_with_root_id(self) -> None:
        llm = _make_llm()
        msgs = [{"role": "user", "content": "follow-up"}]
        payload = llm._build_payload_from_converted(
            msgs, stream=False, root_id=999
        )
        assert payload["root_id"] == 999
        assert payload["messages"] == msgs


# ------------------------------------------------------------------
# Conversation tracking  (JSON-only protocol)
# ------------------------------------------------------------------


def _make_execute_response(
    output: str,
    root_id: int,
    response_id: int | None = None,
) -> dict:
    rid = response_id or root_id
    return {
        "template_id": "8794",
        "responses": [
            {
                "id": rid,
                "status": "succeeded",
                "input": "",
                "output": output,
                "credits": 0.005,
                "root_id": root_id,
                "created_at": "2025-01-05T19:35:21.000000Z",
                "updated_at": "2025-01-05T19:35:23.000000Z",
                "template_id": 8794,
            }
        ],
    }


class TestConversationTracking:
    def test_first_call_includes_developer_and_user(self) -> None:
        llm = _make_llm(wait_execution=True)
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Hello!"), 5001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("hi")

        p = captured_payloads[0]
        assert "root_id" not in p
        assert p["messages"][0]["role"] == "developer"
        assert p["messages"][-1] == {"role": "user", "content": "hi"}

    def test_continuation_sends_delta_with_root_id(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("World!"), 5001, 5002),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        assert len(captured_payloads) == 2

        p1 = captured_payloads[0]
        assert "root_id" not in p1

        p2 = captured_payloads[1]
        assert p2["root_id"] == 5001
        assert p2["messages"] == [{"role": "user", "content": "B"}]

    def test_edited_history_sends_all_without_root_id(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("New conversation!"), 6001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="EDITED response"),
                HumanMessage(content="B"),
            ])

        p2 = captured_payloads[1]
        assert "root_id" not in p2
        roles = [m["role"] for m in p2["messages"]]
        assert roles == ["developer", "user", "assistant", "user"]
        assert p2["messages"][-1]["content"] == "B"

    def test_three_step_continuation(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            outputs = {1: _jc("B"), 2: _jc("D"), 3: _jc("F")}
            return httpx.Response(
                200,
                json=_make_execute_response(
                    outputs[call_count], 5001, 5000 + call_count
                ),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="B"),
                HumanMessage(content="C"),
            ])
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="B"),
                HumanMessage(content="C"),
                AIMessage(content="D"),
                HumanMessage(content="E"),
            ])

        assert "root_id" not in captured_payloads[0]
        assert captured_payloads[1]["root_id"] == 5001
        assert captured_payloads[1]["messages"] == [
            {"role": "user", "content": "C"}
        ]
        assert captured_payloads[2]["root_id"] == 5001
        assert captured_payloads[2]["messages"] == [
            {"role": "user", "content": "E"}
        ]

    def test_continuation_with_tool_calls_sends_delta(self) -> None:
        @tool
        def dummy_tool(q: str) -> str:
            """Test tool."""
            return "tool-ok"

        llm = _make_llm(wait_execution=True)
        llm_tools = llm.bind_tools([dummy_tool])
        call_count = 0
        captured_payloads: list[dict] = []

        raw_api = json.dumps({
            "content": "",
            "commands": [{"name": "dummy_tool", "arguments": {"q": "hello"}}],
        })

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(raw_api, 5001),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Final."), 5001, 5002),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            first = llm_tools.invoke("Use the tool")
            assert first.tool_calls
            tid = first.tool_calls[0]["id"]
            llm_tools.invoke([
                HumanMessage(content="Use the tool"),
                AIMessage(content="", tool_calls=first.tool_calls),
                ToolMessage(
                    content="tool-ok",
                    tool_call_id=tid,
                    name="dummy_tool",
                ),
            ])

        assert len(captured_payloads) == 2
        assert "root_id" not in captured_payloads[0]
        assert captured_payloads[1]["root_id"] == 5001
        delta = captured_payloads[1]["messages"]
        assert len(delta) == 1
        assert delta[0]["role"] == "user"
        assert "[Command Result]" in delta[0]["content"]


class TestConversationTrackingDisabled:
    def test_disabled_always_sends_full_history(self) -> None:
        llm = _make_llm(wait_execution=True, track_conversations=False)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Hello!"), 5001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        p2 = captured_payloads[1]
        assert "root_id" not in p2
        roles = [m["role"] for m in p2["messages"]]
        assert "developer" in roles
        assert roles.count("user") == 2


class TestResetConversations:
    def test_reset_clears_cache(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Hello!"), 5001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            llm.reset_conversations()
            llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        p2 = captured_payloads[1]
        assert "root_id" not in p2


class TestConversationCacheLRU:
    def test_eviction_when_exceeding_max(self) -> None:
        llm = _make_llm(wait_execution=True, max_tracked_conversations=2)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            return httpx.Response(
                200,
                json=_make_execute_response(
                    _jc(f"Response {call_count}"), call_count * 1000
                ),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("conv1")
            llm.invoke("conv2")
            llm.invoke("conv3")

            llm.invoke([
                HumanMessage(content="conv1"),
                AIMessage(content="Response 1"),
                HumanMessage(content="follow-up"),
            ])

        p4 = captured_payloads[3]
        assert "root_id" not in p4


class TestConversationTrackingThreadSafety:
    def test_concurrent_updates(self) -> None:
        llm = _make_llm(wait_execution=True)
        errors: list[Exception] = []

        def fake_post(self_client, url, **kwargs):
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("ok"), 5001),
                request=httpx.Request("POST", url),
            )

        def invoke_in_thread(msg: str) -> None:
            try:
                llm.invoke(msg)
            except Exception as e:
                errors.append(e)

        with patch.object(httpx.Client, "post", fake_post):
            threads = [
                threading.Thread(target=invoke_in_thread, args=(f"msg-{i}",))
                for i in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert errors == []


class TestConversationTrackingMultipleConversations:
    def test_parallel_conversations_tracked_independently(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            root_ids = {1: 1000, 2: 2000, 3: 1000, 4: 2000}
            return httpx.Response(
                200,
                json=_make_execute_response(
                    _jc(f"R{call_count}"), root_ids[call_count]
                ),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("conv-A-msg1")
            llm.invoke("conv-B-msg1")

            llm.invoke([
                HumanMessage(content="conv-A-msg1"),
                AIMessage(content="R1"),
                HumanMessage(content="conv-A-msg2"),
            ])
            llm.invoke([
                HumanMessage(content="conv-B-msg1"),
                AIMessage(content="R2"),
                HumanMessage(content="conv-B-msg2"),
            ])

        p3 = captured_payloads[2]
        assert p3["root_id"] == 1000
        assert p3["messages"] == [{"role": "user", "content": "conv-A-msg2"}]

        p4 = captured_payloads[3]
        assert p4["root_id"] == 2000
        assert p4["messages"] == [{"role": "user", "content": "conv-B-msg2"}]


class TestConversationTrackingAsync:
    @pytest.mark.asyncio
    async def test_async_continuation_sends_delta(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        async def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("World!"), 5001, 5002),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.AsyncClient, "post", fake_post):
            await llm.ainvoke("A")
            await llm.ainvoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        p1 = captured_payloads[0]
        assert "root_id" not in p1

        p2 = captured_payloads[1]
        assert p2["root_id"] == 5001
        assert p2["messages"] == [{"role": "user", "content": "B"}]


class TestConversationTrackingStream:
    def test_stream_continuation_sends_delta(self) -> None:
        llm = _make_llm(wait_execution=True)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("World!"), 5001, 5002),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            list(llm.stream("A"))
            list(llm.stream([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ]))

        p1 = captured_payloads[0]
        assert "root_id" not in p1

        p2 = captured_payloads[1]
        assert p2["root_id"] == 5001
        assert p2["messages"] == [{"role": "user", "content": "B"}]


# ------------------------------------------------------------------
# Continuation fallback on failure
# ------------------------------------------------------------------


class TestContinuationFallback:
    def test_http_error_triggers_fallback(self) -> None:
        llm = _make_llm(wait_execution=True, max_retries=1)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            if call_count == 2:
                raise httpx.TransportError("connection lost")
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Recovered!"), 6001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            result = llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        assert call_count == 3

        p2 = captured_payloads[1]
        assert p2["root_id"] == 5001
        assert p2["messages"] == [{"role": "user", "content": "B"}]

        p3 = captured_payloads[2]
        assert "root_id" not in p3
        roles = [m["role"] for m in p3["messages"]]
        assert roles == ["developer", "user", "assistant", "user"]

        assert result.content == "Recovered!"
        assert result.response_metadata["tess_root_id"] == 6001

    def test_empty_response_triggers_fallback(self) -> None:
        llm = _make_llm(wait_execution=True, max_retries=1)
        call_count = 0
        captured_payloads: list[dict] = []

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_payloads.append(kwargs.get("json", {}))
            if call_count == 1:
                return httpx.Response(
                    200,
                    json=_make_execute_response(_jc("Hello!"), 5001),
                    request=httpx.Request("POST", url),
                )
            if call_count == 2:
                return httpx.Response(
                    200,
                    json=_make_execute_response("", 5001, 5002),
                    request=httpx.Request("POST", url),
                )
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("Recovered!"), 7001),
                request=httpx.Request("POST", url),
            )

        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke("A")
            result = llm.invoke([
                HumanMessage(content="A"),
                AIMessage(content="Hello!"),
                HumanMessage(content="B"),
            ])

        p2 = captured_payloads[1]
        assert p2["root_id"] == 5001

        p3 = captured_payloads[2]
        assert "root_id" not in p3

        assert result.response_metadata["tess_root_id"] == 7001


# ------------------------------------------------------------------
# Multimodal file extraction
# ------------------------------------------------------------------

SAMPLE_B64 = base64.b64encode(b"hello world").decode()
SAMPLE_BYTES = b"hello world"
SAMPLE_HASH = hashlib.sha256(SAMPLE_BYTES).hexdigest()

UPLOAD_RESPONSE_COMPLETED = {
    "id": 55,
    "object": "file",
    "bytes": 11,
    "created_at": "2025-01-05T22:26:27+00:00",
    "filename": "test.pdf",
    "credits": 0,
    "status": "completed",
}

UPLOAD_RESPONSE_WAITING = {
    "id": 72,
    "object": "file",
    "bytes": 11,
    "created_at": "2025-01-05T22:26:27+00:00",
    "filename": "test.csv",
    "credits": 0,
    "status": "waiting",
}

FILE_POLL_COMPLETED = {
    "id": 72,
    "object": "file",
    "bytes": 11,
    "created_at": "2025-01-05T22:26:27+00:00",
    "filename": "test.csv",
    "credits": 1.5,
    "status": "completed",
}

FILE_POLL_FAILED = {
    "id": 72,
    "object": "file",
    "bytes": 11,
    "created_at": "2025-01-05T22:26:27+00:00",
    "filename": "test.csv",
    "credits": 0,
    "status": "failed",
}


class TestConvertMessagesWithFiles:
    def test_text_only_message(self) -> None:
        msgs = [HumanMessage(content="hello")]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)
        assert converted == [{"role": "user", "content": "hello"}]
        assert file_refs == []

    def test_image_block(self) -> None:
        msgs = [HumanMessage(content=[
            {"type": "text", "text": "Describe this"},
            {
                "type": "image",
                "mimeType": "image/png",
                "data": SAMPLE_B64,
                "metadata": {"name": "photo.png"},
            },
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Describe this"

        assert len(file_refs) == 1
        assert file_refs[0].data == SAMPLE_BYTES
        assert file_refs[0].mime_type == "image/png"
        assert file_refs[0].file_id is None
        assert file_refs[0].needs_upload is True

    def test_file_block_pdf(self) -> None:
        msgs = [HumanMessage(content=[
            {"type": "text", "text": "Summarize this PDF"},
            {
                "type": "file",
                "mimeType": "application/pdf",
                "data": SAMPLE_B64,
                "metadata": {"filename": "paper.pdf"},
            },
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert converted[0]["content"] == "Summarize this PDF"
        assert len(file_refs) == 1
        assert file_refs[0].data == SAMPLE_BYTES
        assert file_refs[0].mime_type == "application/pdf"
        assert file_refs[0].needs_upload is True

    def test_url_block(self) -> None:
        msgs = [HumanMessage(content=[
            {"type": "text", "text": "Check this"},
            {
                "type": "url",
                "url": "https://example.com/data.csv",
                "mimeType": "text/csv",
            },
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert converted[0]["content"] == "Check this"
        assert len(file_refs) == 1
        assert file_refs[0].url == "https://example.com/data.csv"
        assert file_refs[0].mime_type == "text/csv"
        assert file_refs[0].data is None
        assert file_refs[0].needs_upload is True

    def test_tess_ai_block(self) -> None:
        msgs = [HumanMessage(content=[
            {"type": "text", "text": "Summarize"},
            {"type": "tess_ai", "file_id": 99},
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert converted[0]["content"] == "Summarize"
        assert len(file_refs) == 1
        assert file_refs[0].file_id == 99
        assert file_refs[0].needs_upload is False

    def test_mixed_text_and_files(self) -> None:
        msgs = [HumanMessage(content=[
            {"type": "text", "text": "Line 1"},
            {
                "type": "image",
                "mimeType": "image/png",
                "data": SAMPLE_B64,
            },
            {"type": "text", "text": "Line 2"},
            {"type": "tess_ai", "file_id": 42},
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert converted[0]["content"] == "Line 1\nLine 2"
        assert len(file_refs) == 2
        assert file_refs[0].mime_type == "image/png"
        assert file_refs[1].file_id == 42

    def test_multiple_messages_with_files(self) -> None:
        msgs = [
            HumanMessage(content=[
                {"type": "text", "text": "First"},
                {
                    "type": "file",
                    "mimeType": "application/pdf",
                    "data": SAMPLE_B64,
                },
            ]),
            AIMessage(content="Got it"),
            HumanMessage(content=[
                {"type": "text", "text": "Second"},
                {
                    "type": "file",
                    "mimeType": "text/csv",
                    "data": SAMPLE_B64,
                },
            ]),
        ]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)

        assert len(converted) == 3
        assert converted[0]["content"] == "First"
        assert converted[2]["content"] == "Second"
        assert len(file_refs) == 2
        assert file_refs[0].message_index == 0
        assert file_refs[1].message_index == 2

    def test_only_file_no_text(self) -> None:
        msgs = [HumanMessage(content=[
            {
                "type": "file",
                "mimeType": "application/pdf",
                "data": SAMPLE_B64,
            },
        ])]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)
        assert converted[0]["content"] == "[See attached files]"
        assert len(file_refs) == 1

    def test_merge_consecutive_user_messages(self) -> None:
        msgs = [
            HumanMessage(content=[
                {"type": "text", "text": "Part A"},
                {
                    "type": "image",
                    "mimeType": "image/png",
                    "data": SAMPLE_B64,
                },
            ]),
            HumanMessage(content="Part B"),
        ]
        converted, file_refs = ChatTessAI._convert_messages_with_files(msgs)
        assert len(converted) == 1
        assert "Part A" in converted[0]["content"]
        assert "Part B" in converted[0]["content"]
        assert len(file_refs) == 1


class TestFileRef:
    def test_content_hash(self) -> None:
        ref = FileRef(data=SAMPLE_BYTES, mime_type="application/pdf")
        assert ref.content_hash == SAMPLE_HASH

    def test_content_hash_none_when_no_data(self) -> None:
        ref = FileRef(url="https://example.com/file.pdf")
        assert ref.content_hash is None

    def test_extension_known_mime(self) -> None:
        ref = FileRef(data=b"x", mime_type="application/pdf")
        assert ref.extension == ".pdf"

    def test_extension_unknown_mime(self) -> None:
        ref = FileRef(data=b"x", mime_type="application/x-unknown-thing")
        assert ref.extension  # falls back to mimetypes or .bin

    def test_upload_filename(self) -> None:
        ref = FileRef(data=SAMPLE_BYTES, mime_type="text/csv")
        assert ref.upload_filename.endswith(".csv")
        assert ref.upload_filename.startswith(SAMPLE_HASH[:12])

    def test_needs_upload_true(self) -> None:
        ref = FileRef(data=b"x", mime_type="text/csv")
        assert ref.needs_upload is True

    def test_needs_upload_false(self) -> None:
        ref = FileRef(file_id=55)
        assert ref.needs_upload is False


class TestUploadAndProcessFile:
    def test_upload_completed_immediately(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_upload) as mock_post:
            with llm._sync_client() as client:
                fid = llm._upload_and_process_file(client, b"data", "test.pdf")

        assert fid == 55
        mock_post.assert_called_once()

    def test_upload_with_polling(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_WAITING,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )
        mock_poll = httpx.Response(
            200,
            json=FILE_POLL_COMPLETED,
            request=httpx.Request("GET", f"{BASE_URL}/files/72"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_upload), \
             patch.object(httpx.Client, "get", return_value=mock_poll):
            with llm._sync_client() as client:
                fid = llm._upload_and_process_file(client, b"data", "test.csv")

        assert fid == 72

    def test_upload_failed(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_WAITING,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )
        mock_poll = httpx.Response(
            200,
            json=FILE_POLL_FAILED,
            request=httpx.Request("GET", f"{BASE_URL}/files/72"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_upload), \
             patch.object(httpx.Client, "get", return_value=mock_poll):
            with llm._sync_client() as client:
                with pytest.raises(RuntimeError, match="processing failed"):
                    llm._upload_and_process_file(client, b"data", "test.csv")

    def test_upload_dedup(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_upload):
            with llm._sync_client() as client:
                fid1 = llm._upload_and_process_file(client, b"same", "a.pdf")
                fid2 = llm._upload_and_process_file(client, b"same", "b.pdf")

        assert fid1 == fid2 == 55


class TestResolveFileIds:
    def test_tess_ai_ref_no_upload(self) -> None:
        llm = _make_llm()
        refs = [FileRef(file_id=99)]

        with llm._sync_client() as client:
            result = llm._resolve_file_ids(client, refs)

        assert result == [99]

    def test_base64_cache_hit(self) -> None:
        llm = _make_llm()
        llm._file_cache[SAMPLE_HASH] = 55
        refs = [FileRef(data=SAMPLE_BYTES, mime_type="application/pdf")]

        with llm._sync_client() as client:
            result = llm._resolve_file_ids(client, refs)

        assert result == [55]

    def test_base64_cache_miss_uploads(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        refs = [FileRef(data=SAMPLE_BYTES, mime_type="application/pdf")]

        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_upload):
            with llm._sync_client() as client:
                result = llm._resolve_file_ids(client, refs)

        assert result == [55]
        assert llm._file_cache[SAMPLE_HASH] == 55

    def test_url_downloads_then_uploads(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        refs = [FileRef(url="https://example.com/data.csv", mime_type="text/csv")]

        mock_download = httpx.Response(
            200,
            content=SAMPLE_BYTES,
            headers={"content-type": "text/csv"},
            request=httpx.Request("GET", "https://example.com/data.csv"),
        )
        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )

        with patch.object(httpx.Client, "get", return_value=mock_download), \
             patch.object(httpx.Client, "post", return_value=mock_upload):
            with llm._sync_client() as client:
                result = llm._resolve_file_ids(client, refs)

        assert result == [55]
        assert SAMPLE_HASH in llm._file_cache

    def test_mixed_refs(self) -> None:
        llm = _make_llm(polling_interval=0.01)
        refs = [
            FileRef(file_id=99),
            FileRef(data=SAMPLE_BYTES, mime_type="application/pdf"),
        ]
        llm._file_cache[SAMPLE_HASH] = 55

        with llm._sync_client() as client:
            result = llm._resolve_file_ids(client, refs)

        assert result == [99, 55]

    def test_static_file_ids_merged(self) -> None:
        result = ChatTessAI._merge_file_ids([10, 20], [20, 30])
        assert result == [10, 20, 30]

    def test_merge_file_ids_empty(self) -> None:
        result = ChatTessAI._merge_file_ids(None, [])
        assert result is None


class TestGenerateWithFiles:
    def test_invoke_with_document_block(self) -> None:
        llm = _make_llm(polling_interval=0.01)

        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )
        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        captured_payloads = []
        original_post = httpx.Client.post

        def fake_post(self_client, url, **kwargs):
            if "/files" in url:
                return mock_upload
            captured_payloads.append(kwargs.get("json", {}))
            return mock_execute

        with patch.object(httpx.Client, "post", fake_post):
            result = llm.invoke([HumanMessage(content=[
                {"type": "text", "text": "Analyze this PDF"},
                {
                    "type": "file",
                    "mimeType": "application/pdf",
                    "data": SAMPLE_B64,
                },
            ])])

        assert isinstance(result, AIMessage)
        assert result.content == "Hello! How can I help you?"
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["file_ids"] == [55]

    def test_invoke_with_multiple_files_across_turns(self) -> None:
        llm = _make_llm(polling_interval=0.01, track_conversations=False)

        call_count = 0

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            if "/files" in url:
                call_count += 1
                return httpx.Response(
                    200,
                    json={**UPLOAD_RESPONSE_COMPLETED, "id": 50 + call_count},
                    request=httpx.Request("POST", f"{BASE_URL}/files"),
                )
            payload = kwargs.get("json", {})
            captured_payloads.append(payload)
            return httpx.Response(
                200,
                json=_make_execute_response(_jc("OK"), 5001),
                request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
            )

        captured_payloads: list = []
        with patch.object(httpx.Client, "post", fake_post):
            llm.invoke([HumanMessage(content=[
                {"type": "text", "text": "First doc"},
                {"type": "file", "mimeType": "application/pdf", "data": SAMPLE_B64},
            ])])

            other_b64 = base64.b64encode(b"different content").decode()
            llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": "First doc"},
                    {"type": "file", "mimeType": "application/pdf", "data": SAMPLE_B64},
                ]),
                AIMessage(content="OK"),
                HumanMessage(content=[
                    {"type": "text", "text": "Second doc"},
                    {"type": "file", "mimeType": "text/csv", "data": other_b64},
                ]),
            ])

        assert 51 in captured_payloads[0]["file_ids"]
        p2_file_ids = captured_payloads[1]["file_ids"]
        assert 51 in p2_file_ids
        assert 52 in p2_file_ids

    def test_file_ids_not_invalidated_on_error(self) -> None:
        llm = _make_llm(polling_interval=0.01, max_retries=0)

        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )
        mock_error = httpx.Response(
            500,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        mock_success = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        call_count = 0

        def fake_post(self_client, url, **kwargs):
            nonlocal call_count
            if "/files" in url:
                return mock_upload
            call_count += 1
            if call_count == 1:
                return mock_error
            return mock_success

        with patch.object(httpx.Client, "post", fake_post):
            with pytest.raises((httpx.HTTPStatusError, TessAPIError)):
                llm.invoke([HumanMessage(content=[
                    {"type": "text", "text": "Test"},
                    {"type": "file", "mimeType": "application/pdf", "data": SAMPLE_B64},
                ])])

        assert SAMPLE_HASH in llm._file_cache
        assert llm._file_cache[SAMPLE_HASH] == 55


class TestGenerateWithFilesAsync:
    async def test_ainvoke_with_document_block(self) -> None:
        llm = _make_llm(polling_interval=0.01)

        mock_upload = httpx.Response(
            200,
            json=UPLOAD_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/files"),
        )
        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        captured_payloads = []

        async def fake_post(self_client, url, **kwargs):
            if "/files" in url:
                return mock_upload
            captured_payloads.append(kwargs.get("json", {}))
            return mock_execute

        with patch.object(httpx.AsyncClient, "post", fake_post):
            result = await llm.ainvoke([HumanMessage(content=[
                {"type": "text", "text": "Analyze"},
                {
                    "type": "file",
                    "mimeType": "application/pdf",
                    "data": SAMPLE_B64,
                },
            ])])

        assert isinstance(result, AIMessage)
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["file_ids"] == [55]


class TestBuildPayloadFileIds:
    def test_file_ids_param_overrides_static(self) -> None:
        llm = _make_llm(file_ids=[10, 20])
        msgs = [{"role": "user", "content": "hi"}]
        payload = llm._build_payload_from_converted(msgs, file_ids=[30, 40])
        assert payload["file_ids"] == [30, 40]

    def test_file_ids_param_none_falls_back_to_static(self) -> None:
        llm = _make_llm(file_ids=[10, 20])
        msgs = [{"role": "user", "content": "hi"}]
        payload = llm._build_payload_from_converted(msgs, file_ids=None)
        assert payload["file_ids"] == [10, 20]

    def test_no_file_ids_at_all(self) -> None:
        llm = _make_llm()
        msgs = [{"role": "user", "content": "hi"}]
        payload = llm._build_payload_from_converted(msgs)
        assert "file_ids" not in payload


# ==================================================================
# Tests: Typed exceptions (TessAPIError hierarchy)
# ==================================================================


from langchain_tessai.exceptions import (
    TessAPIError,
    TessAuthenticationError,
    TessPayloadTooLargeError,
    TessRateLimitError,
    TessServerError,
    TessValidationError,
    raise_for_tess_status,
)


class TestTypedExceptions:
    def test_raise_400_validation_error(self) -> None:
        with pytest.raises(TessValidationError) as exc_info:
            raise_for_tess_status(400, {"error": "Validation failed"})
        assert exc_info.value.status_code == 400
        assert "Validation failed" in str(exc_info.value)

    def test_raise_403_authentication_error(self) -> None:
        with pytest.raises(TessAuthenticationError) as exc_info:
            raise_for_tess_status(403, {"error": "Invalid authentication"})
        assert exc_info.value.status_code == 403

    def test_raise_413_payload_too_large(self) -> None:
        with pytest.raises(TessPayloadTooLargeError):
            raise_for_tess_status(413, {"error": "Payload too large"})

    def test_raise_429_rate_limit_with_retry_after(self) -> None:
        with pytest.raises(TessRateLimitError) as exc_info:
            raise_for_tess_status(429, {"error": "Rate limit exceeded", "retry_after": 30})
        assert exc_info.value.retry_after == 30

    def test_raise_429_default_retry_after(self) -> None:
        with pytest.raises(TessRateLimitError) as exc_info:
            raise_for_tess_status(429, {"error": "Rate limit exceeded"})
        assert exc_info.value.retry_after == 60

    def test_raise_500_server_error(self) -> None:
        with pytest.raises(TessServerError):
            raise_for_tess_status(500, {"error": "Internal server error"})

    def test_raise_502_as_server_error(self) -> None:
        with pytest.raises(TessServerError):
            raise_for_tess_status(502, None)

    def test_no_raise_on_200(self) -> None:
        raise_for_tess_status(200, None)

    def test_all_inherit_from_tess_api_error(self) -> None:
        for cls in (
            TessAuthenticationError,
            TessValidationError,
            TessRateLimitError,
            TessPayloadTooLargeError,
            TessServerError,
        ):
            assert issubclass(cls, TessAPIError)


class TestTypedExceptionsInGenerate:
    def test_403_raises_immediately_no_retry(self) -> None:
        llm = _make_llm(max_retries=3)

        mock_response = httpx.Response(
            403,
            json={"error": "Invalid authentication"},
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        call_count = 0
        original_post = httpx.Client.post

        def counting_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(httpx.Client, "post", counting_post):
            with pytest.raises(TessAuthenticationError):
                llm.invoke("hello")

        assert call_count == 1

    def test_413_raises_immediately_no_retry(self) -> None:
        llm = _make_llm(max_retries=3)

        mock_response = httpx.Response(
            413,
            json={"error": "Payload too large"},
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        call_count = 0

        def counting_post(self_client, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(httpx.Client, "post", counting_post):
            with pytest.raises(TessPayloadTooLargeError):
                llm.invoke("hello")

        assert call_count == 1

    def test_429_retries_with_retry_after(self) -> None:
        llm = _make_llm(max_retries=1)

        mock_429 = httpx.Response(
            429,
            json={"error": "Rate limit exceeded", "retry_after": 0},
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        mock_success = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        calls = []

        def side_effect_post(self_client, url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                return mock_429
            return mock_success

        with patch.object(httpx.Client, "post", side_effect_post):
            result = llm.invoke("hello")

        assert isinstance(result, AIMessage)
        assert len(calls) == 2

    def test_500_uses_exponential_backoff(self) -> None:
        llm = _make_llm(max_retries=1)

        mock_500 = httpx.Response(
            500,
            json={"error": "Internal server error"},
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_500):
            with pytest.raises(TessServerError):
                llm.invoke("hello")


# ==================================================================
# Tests: stop client-side truncation
# ==================================================================


class TestStopSequences:
    def test_apply_stop_sequences_single(self) -> None:
        result = ChatTessAI._apply_stop_sequences("hello world stop here", ["stop"])
        assert result == "hello world "

    def test_apply_stop_sequences_multiple(self) -> None:
        result = ChatTessAI._apply_stop_sequences("aaa bbb ccc", ["bbb", "ccc"])
        assert result == "aaa "

    def test_apply_stop_sequences_none(self) -> None:
        result = ChatTessAI._apply_stop_sequences("hello world", None)
        assert result == "hello world"

    def test_apply_stop_sequences_empty_list(self) -> None:
        result = ChatTessAI._apply_stop_sequences("hello world", [])
        assert result == "hello world"

    def test_apply_stop_sequences_not_found(self) -> None:
        result = ChatTessAI._apply_stop_sequences("hello world", ["xyz"])
        assert result == "hello world"

    def test_stop_in_generate(self) -> None:
        llm = _make_llm()

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": '{"content": "Hello! STOP How can I help?"}',
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            result = llm.invoke("hello", stop=["STOP"])

        assert result.content == "Hello! "


# ==================================================================
# Tests: model_name in response_metadata
# ==================================================================


class TestModelNameMetadata:
    def test_model_name_in_metadata(self) -> None:
        llm = _make_llm(model="gpt-4o")

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            result = llm.invoke("hello")

        assert result.response_metadata["model_name"] == "gpt-4o"

    def test_model_name_in_poll_metadata(self) -> None:
        llm = _make_llm(model="tess-5", wait_execution=True, polling_interval=0.01)

        mock_execute = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_STARTING,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )
        mock_poll = httpx.Response(
            200,
            json=POLL_RESPONSE_SUCCEEDED,
            request=httpx.Request("GET", f"{BASE_URL}/agent-responses/5002"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_execute):
            with patch.object(httpx.Client, "get", return_value=mock_poll):
                result = llm.invoke("hello")

        assert result.response_metadata["model_name"] == "tess-5"


# ==================================================================
# Tests: tool_choice via prompt engineering
# ==================================================================


class TestToolChoicePrompt:
    def test_tool_choice_none_no_extra_instruction(self) -> None:
        llm = _make_llm()

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        bound = llm.bind_tools([get_weather], tool_choice=None)
        assert bound.kwargs.get("tool_choice") is None

    def test_tool_choice_required_in_prompt(self) -> None:
        llm = _make_llm()

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "Checking weather",
                        "commands": [{"name": "get_weather", "arguments": {"city": "SP"}}],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        captured_payloads = []

        def fake_post(self_client, url, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            return mock_response

        with patch.object(httpx.Client, "post", fake_post):
            bound = llm.bind_tools([get_weather], tool_choice="required")
            bound.invoke("What's the weather in SP?")

        developer_msg = captured_payloads[0]["messages"][0]
        assert "MUST execute at least one command" in developer_msg["content"]

    def test_tool_choice_none_instruction_in_prompt(self) -> None:
        llm = _make_llm()

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        captured_payloads = []

        def fake_post(self_client, url, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            return mock_response

        with patch.object(httpx.Client, "post", fake_post):
            bound = llm.bind_tools([get_weather], tool_choice="none")
            bound.invoke("Just chat, no tools")

        developer_msg = captured_payloads[0]["messages"][0]
        assert "Do NOT execute any commands" in developer_msg["content"]

    def test_tool_choice_specific_tool(self) -> None:
        llm = _make_llm()

        @tool
        def get_weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "",
                        "commands": [{"name": "get_weather", "arguments": {"city": "SP"}}],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        captured_payloads = []

        def fake_post(self_client, url, **kwargs):
            captured_payloads.append(kwargs.get("json", {}))
            return mock_response

        with patch.object(httpx.Client, "post", fake_post):
            bound = llm.bind_tools(
                [get_weather],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            )
            bound.invoke("Weather?")

        developer_msg = captured_payloads[0]["messages"][0]
        assert 'MUST execute the command "get_weather"' in developer_msg["content"]


# ==================================================================
# Tests: with_structured_output for dict schemas
# ==================================================================


class TestStructuredOutputDict:
    def test_with_dict_schema(self) -> None:
        llm = _make_llm()

        schema = {
            "title": "WeatherReport",
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "temp": {"type": "number"},
            },
            "required": ["city", "temp"],
        }

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "",
                        "commands": [{
                            "name": "WeatherReport",
                            "arguments": {"city": "SP", "temp": 25},
                        }],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            chain = llm.with_structured_output(schema)
            result = chain.invoke("Weather in SP?")

        assert isinstance(result, dict)
        assert result["city"] == "SP"
        assert result["temp"] == 25

    def test_with_dict_schema_include_raw(self) -> None:
        llm = _make_llm()

        schema = {"title": "Info", "type": "object", "properties": {"x": {"type": "string"}}}

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "",
                        "commands": [{"name": "Info", "arguments": {"x": "hello"}}],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            chain = llm.with_structured_output(schema, include_raw=True)
            result = chain.invoke("test")

        assert "raw" in result
        assert "parsed" in result
        assert result["parsed"] == {"x": "hello"}

    def test_with_dict_schema_fallback_name(self) -> None:
        llm = _make_llm()

        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "",
                        "commands": [{"name": "structured_output", "arguments": {"a": 42}}],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            chain = llm.with_structured_output(schema)
            result = chain.invoke("test")

        assert result == {"a": 42}

    def test_with_pydantic_still_works(self) -> None:
        from pydantic import BaseModel

        class Weather(BaseModel):
            city: str
            temp: float

        llm = _make_llm()

        mock_response = httpx.Response(
            200,
            json={
                "template_id": "8794",
                "responses": [{
                    "id": 5001,
                    "status": "succeeded",
                    "output": json.dumps({
                        "content": "",
                        "commands": [{"name": "Weather", "arguments": {"city": "SP", "temp": 25.0}}],
                    }),
                    "credits": 0.006,
                    "root_id": 5001,
                    "template_id": 8794,
                }],
            },
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            chain = llm.with_structured_output(Weather)
            result = chain.invoke("Weather?")

        assert isinstance(result, Weather)
        assert result.city == "SP"

    def test_invalid_schema_type_raises(self) -> None:
        llm = _make_llm()
        with pytest.raises(TypeError, match="dict"):
            llm.with_structured_output(42)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# Token counting (tiktoken)
# ------------------------------------------------------------------


class TestTokenCounting:
    def test_get_num_tokens_gpt_model(self) -> None:
        llm = _make_llm(model="gpt-4o")
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        text = "Hello, how are you?"
        assert llm.get_num_tokens(text) == len(enc.encode(text))

    def test_get_num_tokens_non_gpt_model(self) -> None:
        llm = _make_llm(model="tess-5")
        import tiktoken
        enc = tiktoken.encoding_for_model("text-davinci-003")
        text = "Hello, how are you?"
        assert llm.get_num_tokens(text) == len(enc.encode(text))

    def test_gpt_and_non_gpt_use_different_encodings(self) -> None:
        gpt_llm = _make_llm(model="gpt-4o-mini")
        tess_llm = _make_llm(model="tess-5")
        text = "Some sample text for testing token counts."
        gpt_tokens = gpt_llm.get_num_tokens(text)
        tess_tokens = tess_llm.get_num_tokens(text)
        assert isinstance(gpt_tokens, int)
        assert isinstance(tess_tokens, int)
        assert gpt_tokens > 0
        assert tess_tokens > 0

    def test_get_num_tokens_from_messages(self) -> None:
        llm = _make_llm(model="tess-5")
        msgs = [
            SystemMessage(content="You are a helper."),
            HumanMessage(content="Hello!"),
        ]
        count = llm.get_num_tokens_from_messages(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_usage_metadata_in_generate(self) -> None:
        llm = _make_llm(wait_execution=True)

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.Client, "post", return_value=mock_response):
            result = llm.invoke("hello")

        assert isinstance(result, AIMessage)
        assert result.usage_metadata is not None
        assert "input_tokens" in result.usage_metadata
        assert "output_tokens" in result.usage_metadata
        assert "total_tokens" in result.usage_metadata
        assert result.usage_metadata["input_tokens"] > 0
        assert result.usage_metadata["output_tokens"] > 0
        assert result.usage_metadata["total_tokens"] == (
            result.usage_metadata["input_tokens"]
            + result.usage_metadata["output_tokens"]
        )

    @pytest.mark.asyncio
    async def test_usage_metadata_in_agenerate(self) -> None:
        llm = _make_llm(wait_execution=True)

        mock_response = httpx.Response(
            200,
            json=EXECUTE_RESPONSE_COMPLETED,
            request=httpx.Request("POST", f"{BASE_URL}/agents/{AGENT_ID}/execute"),
        )

        with patch.object(httpx.AsyncClient, "post", return_value=mock_response):
            result = await llm.ainvoke("hello")

        assert isinstance(result, AIMessage)
        assert result.usage_metadata is not None
        assert result.usage_metadata["input_tokens"] > 0
        assert result.usage_metadata["output_tokens"] > 0
        assert result.usage_metadata["total_tokens"] == (
            result.usage_metadata["input_tokens"]
            + result.usage_metadata["output_tokens"]
        )

    @pytest.mark.asyncio
    async def test_usage_metadata_in_astream(self) -> None:
        llm = _make_llm()

        json_output = '{"content": "Stream hello"}'
        events = [
            {"id": 300, "status": "running", "output": json_output, "error": None,
             "credits": None, "root_id": 300, "template_id": 10},
            {"id": 300, "status": "completed", "output": "", "error": None,
             "credits": 0.003, "root_id": 300, "template_id": 10},
        ]

        class FakeAsyncStreamResponse:
            def __init__(self) -> None:
                self.status_code = 200

            def raise_for_status(self) -> None:
                pass

            async def aiter_lines(self):
                for line in _sse_lines(events):
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeAsyncClient:
            def stream(self, *args, **kwargs):
                return FakeAsyncStreamResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with patch.object(llm, "_async_client", return_value=FakeAsyncClient()):
            chunks = []
            async for chunk in llm.astream("hello"):
                chunks.append(chunk)

        final_chunks = [c for c in chunks if c.usage_metadata]
        assert len(final_chunks) >= 1
        usage = final_chunks[-1].usage_metadata
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
