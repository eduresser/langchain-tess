"""Tess AI chat model integration for LangChain."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import httpx
import tiktoken
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from langchain_tess.exceptions import (
    TessAPIError,
    TessAuthenticationError,
    TessPayloadTooLargeError,
    TessRateLimitError,
    raise_for_tess_status,
)
from langchain_tess.tool_calling import (
    IncrementalJsonContentExtractor,
    ToolCallParseError,
    build_json_prompt,
    has_trailing_content,
    parse_json_response,
)


logger = logging.getLogger(__name__)


MIME_TO_EXTENSION: Dict[str, str] = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "image/tiff": ".tiff",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/quicktime": ".mov",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/x-python": ".py",
    "application/javascript": ".js",
    "text/yaml": ".yaml",
}


@dataclass
class FileRef:
    """Reference to a file extracted from a multimodal LangChain message.

    Supports three ``source.type`` variants:
    - ``base64``: raw bytes with a MIME type (Anthropic-style ``document`` block).
    - ``url``: an external URL to be downloaded before upload.
    - ``tess_ai``: a pre-existing Tess ``file_id`` (no upload needed).
    """

    mime_type: str = ""
    data: Optional[bytes] = None
    url: Optional[str] = None
    file_id: Optional[int] = None
    message_index: int = 0

    @property
    def needs_upload(self) -> bool:
        return self.file_id is None

    @property
    def content_hash(self) -> Optional[str]:
        if self.data is not None:
            return hashlib.sha256(self.data).hexdigest()
        return None

    @property
    def extension(self) -> str:
        ext = MIME_TO_EXTENSION.get(self.mime_type)
        if ext:
            return ext
        guessed = mimetypes.guess_extension(self.mime_type)
        return guessed or ".bin"

    @property
    def upload_filename(self) -> str:
        h = self.content_hash
        prefix = h[:12] if h else "file"
        return f"{prefix}{self.extension}"


_FILE_TERMINAL_STATUSES = frozenset({"completed", "failed"})


class _EmptyResponseError(Exception):
    """Internal sentinel for empty API responses inside the retry loop."""


class ChatTessAI(BaseChatModel):
    """Chat model that uses the Tess AI API.

    Setup:
        Install ``langchain-tess`` and set the ``TESS_API_KEY`` environment variable.

        .. code-block:: bash

            pip install langchain-tess
            export TESS_API_KEY="your-api-key"

    Usage:
        .. code-block:: python

            from langchain_tess import ChatTessAI

            llm = ChatTessAI(
                agent_id=8794,
                model="tess-5",
                temperature=0.5,
            )
            response = llm.invoke("Hello!")
            print(response.content)

    Streaming:
        .. code-block:: python

            for chunk in llm.stream("Tell me a story"):
                print(chunk.content, end="")

    Tool calling (prompt-based):
        .. code-block:: python

            from langchain_core.tools import tool

            @tool
            def get_weather(city: str) -> str:
                \"\"\"Get the weather for a city.\"\"\"
                return f"Sunny in {city}"

            llm_with_tools = llm.bind_tools([get_weather])
            response = llm_with_tools.invoke("What's the weather in SP?")
            print(response.tool_calls)
    """

    api_key: SecretStr = Field(
        default=None,
        alias="tess_api_key",
        description="Tess AI API key. Can also be set via TESS_API_KEY env var.",
    )
    agent_id: int = Field(description="The Tess AI agent ID to use.")
    model: str = Field(default="tess-5", description="Model name to use.")
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    tools: str = Field(
        default="no-tools",
        description="Tess tools: 'no-tools', 'internet', 'twitter', 'wikipedia', etc.",
    )
    workspace_id: int = Field(description="Tess workspace ID.")
    base_url: str = Field(default="https://api.tess.im")
    timeout: int = Field(default=120, description="Request timeout in seconds.")
    max_retries: int = Field(default=5)
    wait_execution: bool = Field(
        default=True,
        description=(
            "Wait for the execution to complete (API has a 100s timeout). "
            "If False, uses polling via GET /agent-responses/{id}."
        ),
    )
    polling_interval: float = Field(
        default=5.0,
        description="Seconds between polling requests when the API returns before completion.",
    )
    file_ids: Optional[List[int]] = Field(
        default=None, description="File IDs to attach to the execution."
    )
    track_conversations: bool = Field(
        default=True,
        description=(
            "Automatically reuse root_id to continue Tess conversations. "
            "When enabled, only new (delta) messages are sent if the "
            "message history is an unmodified extension of a previous call."
        ),
    )
    max_tracked_conversations: int = Field(
        default=100,
        description="Maximum number of conversations kept in the LRU cache.",
    )

    _conversation_cache: Dict[str, Tuple[int, int]] = PrivateAttr(
        default_factory=dict
    )
    _cache_order: List[str] = PrivateAttr(default_factory=list)
    _file_cache: Dict[str, int] = PrivateAttr(default_factory=dict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    model_config = ConfigDict(populate_by_name=True)

    def __init__(self, **kwargs: Any) -> None:
        if "tess_api_key" not in kwargs and "api_key" not in kwargs:
            import os

            env_key = os.environ.get("TESS_API_KEY")
            if env_key:
                kwargs["tess_api_key"] = env_key
        if "api_key" in kwargs and "tess_api_key" not in kwargs:
            kwargs["tess_api_key"] = kwargs.pop("api_key")
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Conversation tracking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_messages(messages: List[dict]) -> str:
        serialized = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _find_conversation(
        self, converted: List[dict]
    ) -> Tuple[Optional[int], int]:
        """Return ``(root_id, prefix_length)`` for the longest known prefix,
        or ``(None, 0)`` when no tracked conversation matches."""
        with self._lock:
            for k in range(len(converted) - 1, 0, -1):
                h = self._hash_messages(converted[:k])
                if h in self._conversation_cache:
                    root_id, _ = self._conversation_cache[h]
                    return root_id, k
        return None, 0

    def _update_conversation_cache(
        self, full_messages: List[dict], root_id: int
    ) -> None:
        h = self._hash_messages(full_messages)
        with self._lock:
            if h in self._conversation_cache:
                self._cache_order.remove(h)
            self._conversation_cache[h] = (root_id, len(full_messages))
            self._cache_order.append(h)
            while len(self._cache_order) > self.max_tracked_conversations:
                evicted = self._cache_order.pop(0)
                self._conversation_cache.pop(evicted, None)

    def _track_after_response(
        self,
        converted: List[dict],
        assistant_msg: AIMessage,
        metadata: dict,
        raw_output: Optional[str] = None,
    ) -> None:
        """Update the conversation cache after a successful API response.

        When *raw_output* contains **trailing content** after the first
        balanced JSON (i.e. the model hallucinated), caching is skipped
        entirely.  On the next turn :meth:`_find_conversation` will find
        no matching prefix, forcing a fresh conversation that keeps the
        Tess-side history free of hallucinated text.

        When the raw output is clean (no trailing content) or not
        provided, the canonical form is used for hashing so that
        trivial JSON-formatting differences between the model's output
        and Python's :func:`json.dumps` do not break continuation.
        """
        if not self.track_conversations:
            return
        new_root_id = metadata.get("tess_root_id")
        if new_root_id is None:
            return
        if raw_output is not None and has_trailing_content(raw_output):
            return
        response_msg = {
            "role": "assistant",
            "content": self._assistant_message_to_tess_content(assistant_msg),
        }
        full_conversation = converted + [response_msg]
        self._update_conversation_cache(full_conversation, new_root_id)

    def _invalidate_root_id(self, root_id: int) -> None:
        """Remove all cache entries associated with *root_id*."""
        with self._lock:
            to_remove = [
                h
                for h, (rid, _) in self._conversation_cache.items()
                if rid == root_id
            ]
            for h in to_remove:
                del self._conversation_cache[h]
                self._cache_order.remove(h)

    def reset_conversations(self) -> None:
        """Clear the internal conversation cache."""
        with self._lock:
            self._conversation_cache.clear()
            self._cache_order.clear()

    # ------------------------------------------------------------------
    # LangChain interface
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "tess-ai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model,
            "agent_id": self.agent_id,
            "temperature": self.temperature,
            "tools": self.tools,
        }

    # ------------------------------------------------------------------
    # Token counting (tiktoken)
    # ------------------------------------------------------------------

    def _get_encoding(self) -> tiktoken.Encoding:
        if self.model.startswith("gpt-"):
            return tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        return tiktoken.encoding_for_model("text-davinci-003")

    def get_num_tokens(self, text: str) -> int:
        return len(self._get_encoding().encode(text))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        enc = self._get_encoding()
        converted = self._convert_messages(messages)
        total = 0
        for msg in converted:
            total += len(enc.encode(msg.get("content", "")))
            total += len(enc.encode(msg.get("role", "")))
        return total

    # ------------------------------------------------------------------
    # bind_tools / with_structured_output
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[
            Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]
        ],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return super().bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            tool_def = convert_to_openai_tool(schema)
            tool_name = tool_def["function"]["name"]

            llm_with_tool = self.bind_tools([schema], **kwargs)

            def _parse(msg: BaseMessage) -> Any:
                if not isinstance(msg, AIMessage) or not msg.tool_calls:
                    if include_raw:
                        return {"raw": msg, "parsed": None, "parsing_error": None}
                    return None
                for tc in msg.tool_calls:
                    if tc["name"] == tool_name:
                        parsed = schema(**tc["args"])
                        if include_raw:
                            return {
                                "raw": msg,
                                "parsed": parsed,
                                "parsing_error": None,
                            }
                        return parsed
                if include_raw:
                    return {"raw": msg, "parsed": None, "parsing_error": None}
                return None

            return llm_with_tool | _parse  # type: ignore[operator]

        if isinstance(schema, dict):
            tool_name = schema.get("title") or schema.get("name") or "structured_output"
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema.get("description", ""),
                    "parameters": schema,
                },
            }

            llm_with_tool = super().bind(tools=[tool_def], **kwargs)

            def _parse_dict(msg: BaseMessage) -> Any:
                if not isinstance(msg, AIMessage) or not msg.tool_calls:
                    if include_raw:
                        return {"raw": msg, "parsed": None, "parsing_error": None}
                    return None
                for tc in msg.tool_calls:
                    if tc["name"] == tool_name:
                        if include_raw:
                            return {
                                "raw": msg,
                                "parsed": tc["args"],
                                "parsing_error": None,
                            }
                        return tc["args"]
                if include_raw:
                    return {"raw": msg, "parsed": None, "parsing_error": None}
                return None

            return llm_with_tool | _parse_dict  # type: ignore[operator]

        raise TypeError(
            f"schema must be a Pydantic BaseModel class or a dict, got {type(schema)}"
        )

    # ------------------------------------------------------------------
    # Core: synchronous generation (unified retry)
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        bound_tools: Optional[List[dict]] = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        messages = self._inject_json_format_prompt(messages, bound_tools, tool_choice)
        converted, file_refs = self._convert_messages_with_files(messages)

        with self._sync_client() as client:
            dynamic_file_ids = (
                self._resolve_file_ids(client, file_refs) if file_refs else []
            )

            merged_file_ids = self._merge_file_ids(self.file_ids, dynamic_file_ids)

            continuation_root_id: Optional[int] = None
            messages_to_send = converted
            if self.track_conversations:
                found_root_id, prefix_len = self._find_conversation(converted)
                if found_root_id is not None:
                    continuation_root_id = found_root_id
                    messages_to_send = converted[prefix_len:]

            payload = self._build_payload_from_converted(
                messages_to_send,
                stream=False,
                root_id=continuation_root_id,
                file_ids=merged_file_ids,
                **kwargs,
            )

            last_error: Optional[Exception] = None
            last_output: str = ""

            def _discard_continuation() -> None:
                nonlocal continuation_root_id, payload
                if continuation_root_id is None:
                    return
                self._invalidate_root_id(continuation_root_id)
                continuation_root_id = None
                payload = self._build_payload_from_converted(
                    converted, stream=False, file_ids=merged_file_ids, **kwargs
                )

            for attempt in range(self.max_retries + 1):
                try:
                    output, metadata = self._execute_sync(client, payload)
                except (TessAuthenticationError, TessPayloadTooLargeError):
                    raise
                except TessRateLimitError as exc:
                    last_error = exc
                    _discard_continuation()
                    if attempt < self.max_retries:
                        time.sleep(exc.retry_after)
                    continue
                except (
                    TessAPIError,
                    httpx.HTTPStatusError,
                    httpx.TransportError,
                ) as exc:
                    last_error = exc
                    _discard_continuation()
                    if attempt < self.max_retries:
                        time.sleep(min(2**attempt, 8))
                    continue

                if not output.strip():
                    last_error = _EmptyResponseError()
                    last_output = output
                    _discard_continuation()
                    if attempt < self.max_retries:
                        time.sleep(min(2**attempt, 8))
                    continue

                try:
                    msg = self._output_to_assistant_message(
                        output, bound_tools, metadata, stop=stop
                    )
                except ToolCallParseError as exc:
                    last_error = exc
                    last_output = exc.raw_output
                    _discard_continuation()
                    if attempt < self.max_retries:
                        time.sleep(min(2**attempt, 8))
                    continue

                self._track_after_response(
                    converted, msg, metadata, raw_output=output,
                )
                enc = self._get_encoding()
                input_tokens = sum(
                    len(enc.encode(m.get("content", ""))) for m in converted
                )
                output_tokens = len(enc.encode(output))
                msg.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                return ChatResult(generations=[ChatGeneration(message=msg)])

        self._raise_after_retries(last_error, last_output)

    # ------------------------------------------------------------------
    # Core: async generation (unified retry)
    # ------------------------------------------------------------------

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio

        bound_tools: Optional[List[dict]] = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        messages = self._inject_json_format_prompt(messages, bound_tools, tool_choice)
        converted, file_refs = self._convert_messages_with_files(messages)

        async with self._async_client() as client:
            dynamic_file_ids = (
                await self._aresolve_file_ids(client, file_refs)
                if file_refs
                else []
            )

            merged_file_ids = self._merge_file_ids(self.file_ids, dynamic_file_ids)

            continuation_root_id: Optional[int] = None
            messages_to_send = converted
            if self.track_conversations:
                found_root_id, prefix_len = self._find_conversation(converted)
                if found_root_id is not None:
                    continuation_root_id = found_root_id
                    messages_to_send = converted[prefix_len:]

            payload = self._build_payload_from_converted(
                messages_to_send,
                stream=False,
                root_id=continuation_root_id,
                file_ids=merged_file_ids,
                **kwargs,
            )

            last_error: Optional[Exception] = None
            last_output: str = ""

            def _discard_continuation() -> None:
                nonlocal continuation_root_id, payload
                if continuation_root_id is None:
                    return
                self._invalidate_root_id(continuation_root_id)
                continuation_root_id = None
                payload = self._build_payload_from_converted(
                    converted, stream=False, file_ids=merged_file_ids, **kwargs
                )

            for attempt in range(self.max_retries + 1):
                try:
                    output, metadata = await self._execute_async(client, payload)
                except (TessAuthenticationError, TessPayloadTooLargeError):
                    raise
                except TessRateLimitError as exc:
                    last_error = exc
                    _discard_continuation()
                    if attempt < self.max_retries:
                        await asyncio.sleep(exc.retry_after)
                    continue
                except (
                    TessAPIError,
                    httpx.HTTPStatusError,
                    httpx.TransportError,
                ) as exc:
                    last_error = exc
                    _discard_continuation()
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
                    continue

                if not output.strip():
                    last_error = _EmptyResponseError()
                    last_output = output
                    _discard_continuation()
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
                    continue

                try:
                    msg = self._output_to_assistant_message(
                        output, bound_tools, metadata, stop=stop
                    )
                except ToolCallParseError as exc:
                    last_error = exc
                    last_output = exc.raw_output
                    _discard_continuation()
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 8))
                    continue

                self._track_after_response(
                    converted, msg, metadata, raw_output=output,
                )
                enc = await asyncio.to_thread(self._get_encoding)
                input_tokens = sum(
                    len(enc.encode(m.get("content", ""))) for m in converted
                )
                output_tokens = len(enc.encode(output))
                msg.usage_metadata = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                return ChatResult(generations=[ChatGeneration(message=msg)])

        self._raise_after_retries(last_error, last_output)

    @staticmethod
    def _content_to_str(raw: Any) -> str:
        """Flatten multimodal content blocks into a plain string."""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts: list[str] = []
            for block in raw:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(raw)

    @staticmethod
    def _assistant_message_to_tess_content(msg: AIMessage) -> str:
        """Tess ``assistant`` content string for *msg* (must match cache keys).

        Always serializes as JSON to match the JSON-only protocol that the
        model uses, so hash-based conversation tracking stays consistent.
        """
        result: Dict[str, Any] = {
            "content": ChatTessAI._content_to_str(msg.content),
        }
        if msg.tool_calls:
            result["commands"] = [
                {"name": tc["name"], "arguments": tc["args"]}
                for tc in msg.tool_calls
            ]
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _apply_stop_sequences(
        content: str, stop: Optional[List[str]]
    ) -> str:
        """Truncate *content* at the first occurrence of any stop sequence.

        The Tess API does not support a ``stop`` parameter, so this
        implements client-side truncation after receiving the response.
        """
        if not stop:
            return content
        earliest = len(content)
        for seq in stop:
            idx = content.find(seq)
            if idx != -1 and idx < earliest:
                earliest = idx
        return content[:earliest]

    def _output_to_assistant_message(
        self,
        output: str,
        bound_tools: Optional[List[dict]],
        metadata: dict,
        stop: Optional[List[str]] = None,
    ) -> AIMessage:
        """Build the final ``AIMessage`` from raw API *output*.

        Always parses the response as JSON (JSON-only protocol).
        ``bound_tools`` is kept for signature compatibility but the
        parsing path is the same regardless.
        """
        content, tool_calls = parse_json_response(output)
        content = self._apply_stop_sequences(content, stop)
        if tool_calls:
            return AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata=metadata,
            )
        return AIMessage(content=content, response_metadata=metadata)

    # ------------------------------------------------------------------
    # Core: synchronous streaming
    # ------------------------------------------------------------------

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Bypass SSE streaming – Cloudflare's ~100 s proxy timeout kills
        # the long-lived connection (HTTP 524) and leaves a phantom
        # server-side execution that causes 500 on subsequent requests.
        # The non-streaming POST + polling path avoids this entirely.
        result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        gen_msg = result.generations[0].message
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=gen_msg.content,
                tool_calls=getattr(gen_msg, "tool_calls", None) or [],
                response_metadata=gen_msg.response_metadata or {},
            )
        )
        yield chunk

    # ------------------------------------------------------------------
    # Core: async streaming
    # ------------------------------------------------------------------

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        import asyncio

        bound_tools: Optional[List[dict]] = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        messages = self._inject_json_format_prompt(messages, bound_tools, tool_choice)
        converted, file_refs = self._convert_messages_with_files(messages)

        extractor = IncrementalJsonContentExtractor()
        last_metadata: dict = {}

        try:
            async with self._async_client() as client:
                dynamic_file_ids = (
                    await self._aresolve_file_ids(client, file_refs)
                    if file_refs
                    else []
                )
                merged_file_ids = self._merge_file_ids(
                    self.file_ids, dynamic_file_ids
                )

                root_id: Optional[int] = None
                messages_to_send = converted
                if self.track_conversations:
                    found_root_id, prefix_len = self._find_conversation(converted)
                    if found_root_id is not None:
                        root_id = found_root_id
                        messages_to_send = converted[prefix_len:]

                payload = self._build_payload_from_converted(
                    messages_to_send,
                    stream=True,
                    root_id=root_id,
                    file_ids=merged_file_ids,
                    **kwargs,
                )
                async with client.stream(
                    "POST",
                    self._execute_url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for chunk_gen in self._aiter_sse_chunks(
                        response.aiter_lines(), run_manager
                    ):
                        if chunk_gen.message.response_metadata:
                            last_metadata = chunk_gen.message.response_metadata
                        new_content = extractor.feed(chunk_gen.message.content)
                        if new_content:
                            content_chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=new_content)
                            )
                            if run_manager:
                                await run_manager.on_llm_new_token(
                                    new_content, chunk=content_chunk
                                )
                            yield content_chunk
        except (httpx.HTTPStatusError, httpx.TransportError):
            kwargs_for_gen = dict(kwargs)
            if bound_tools is not None:
                kwargs_for_gen["tools"] = bound_tools
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs_for_gen
            )
            gen_msg = result.generations[0].message
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=gen_msg.content,
                    tool_calls=getattr(gen_msg, "tool_calls", None) or [],
                    response_metadata=gen_msg.response_metadata or {},
                    usage_metadata=getattr(gen_msg, "usage_metadata", None),
                )
            )
            yield chunk
            return

        full_text = extractor.get_full_text()
        try:
            gen_msg = self._output_to_assistant_message(
                full_text, bound_tools, last_metadata, stop=stop
            )
        except ToolCallParseError:
            content = extractor.get_extracted_content() or full_text
            content = self._apply_stop_sequences(content, stop)
            meta = {**last_metadata, "tess_parse_degraded": True}
            gen_msg = AIMessage(content=content, response_metadata=meta)

        self._track_after_response(
            converted, gen_msg, last_metadata, raw_output=full_text,
        )

        enc = await asyncio.to_thread(self._get_encoding)
        input_tokens = sum(
            len(enc.encode(m.get("content", ""))) for m in converted
        )
        output_tokens = len(enc.encode(full_text))
        stream_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        final_chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="" if extractor.content_complete else gen_msg.content,
                tool_calls=getattr(gen_msg, "tool_calls", None) or [],
                response_metadata=gen_msg.response_metadata or {},
                usage_metadata=stream_usage,
            )
        )
        if run_manager:
            await run_manager.on_llm_new_token(
                final_chunk.message.content or "", chunk=final_chunk
            )
        yield final_chunk

    # ------------------------------------------------------------------
    # JSON format prompt injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_json_format_prompt(
        messages: List[BaseMessage],
        bound_tools: Optional[List[dict]],
        tool_choice: Optional[Union[dict, str, bool]] = None,
    ) -> List[BaseMessage]:
        """Prepend a developer/system message enforcing JSON-only responses.

        Always injects the JSON protocol prompt.  When *bound_tools* is
        provided, includes command definitions and the ``commands`` field
        documentation.  When *tool_choice* is provided, appends an
        instruction constraining command usage.  If the first message is
        already a SystemMessage, its content is appended after the JSON
        prompt.
        """
        json_prompt = build_json_prompt(bound_tools, tool_choice=tool_choice)

        if messages and isinstance(messages[0], SystemMessage):
            combined = f"{json_prompt}\n\n{messages[0].content}"
            return [SystemMessage(content=combined)] + list(messages[1:])

        return [SystemMessage(content=json_prompt)] + list(messages)

    # ------------------------------------------------------------------
    # Execute helpers (single attempt, no retry)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_json_body(resp: httpx.Response) -> dict | None:
        """Try to parse the response body as JSON, return None on failure."""
        try:
            return resp.json()
        except Exception:
            return None

    def _execute_sync(
        self, client: httpx.Client, payload: dict
    ) -> tuple[str, dict]:
        """POST to the execute endpoint, resolve via polling if needed.

        Returns ``(output_text, metadata_dict)``.
        Raises typed :class:`TessAPIError` subclasses for HTTP errors.
        """
        resp = client.post(
            self._execute_url, json=payload, headers=self._headers
        )
        if resp.status_code >= 400:
            raise_for_tess_status(resp.status_code, self._safe_json_body(resp))
        data = resp.json()

        if self.wait_execution and self._is_completed(data):
            return self._extract_output_and_metadata(data)

        response_id = self._extract_response_id(data)
        poll_result = self._poll_for_result(client, response_id)
        return self._extract_output_and_metadata_from_poll(poll_result)

    async def _execute_async(
        self, client: httpx.AsyncClient, payload: dict
    ) -> tuple[str, dict]:
        msgs = payload.get("messages", [])
        msg_summary = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            msg_summary.append(
                f"  [{role}] len={len(content)}"
                + (f" content={content!r}" if role != "developer" else "")
            )
        logger.debug(
            "TESS REQUEST: url=%s model=%s temp=%s tools=%s "
            "wait_exec=%s n_msgs=%d\n%s",
            self._execute_url,
            payload.get("model"),
            payload.get("temperature"),
            payload.get("tools"),
            payload.get("wait_execution"),
            len(msgs),
            "\n".join(msg_summary),
        )

        resp = await client.post(
            self._execute_url, json=payload, headers=self._headers
        )
        if resp.status_code >= 400:
            body = self._safe_json_body(resp)
            logger.warning(
                "TESS API ERROR %s – body: %s", resp.status_code, body
            )
            raise_for_tess_status(resp.status_code, body)
        data = resp.json()

        if self.wait_execution and self._is_completed(data):
            return self._extract_output_and_metadata(data)

        response_id = self._extract_response_id(data)
        poll_result = await self._apoll_for_result(client, response_id)
        return self._extract_output_and_metadata_from_poll(poll_result)

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Tess API format.

        Tess uses ``"developer"`` instead of ``"system"`` and requires it
        to be the first message.  ``user`` / ``assistant`` messages must
        **strictly alternate** after the optional developer message.

        Since the Tess API has no native tool-call protocol, AIMessages
        that carry ``tool_calls`` are serialised as the JSON the model
        originally produced, and ToolMessages are converted into user
        messages that clearly present the tool result so the model can
        incorporate it in its next reply.

        Consecutive messages with the same role (e.g. multiple ToolMessages)
        are merged into a single message to satisfy the alternation
        requirement.
        """
        converted: List[dict] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role, content = "developer", ChatTessAI._content_to_str(msg.content)
            elif isinstance(msg, HumanMessage):
                role, content = "user", ChatTessAI._content_to_str(msg.content)
            elif isinstance(msg, AIMessage):
                role, content = "assistant", ChatTessAI._assistant_message_to_tess_content(msg)
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None) or "unknown_tool"
                role = "user"
                content = (
                    f'[Command Result] The command "{tool_name}" returned:\n'
                    f"{ChatTessAI._content_to_str(msg.content)}\n\n"
                    f'[Respond with a JSON object: {{"content": "your response"}}]'
                )
            else:
                role, content = "user", ChatTessAI._content_to_str(msg.content)

            if converted and converted[-1]["role"] == role and role != "developer":
                converted[-1]["content"] += f"\n\n{content}"
            else:
                converted.append({"role": role, "content": content})
        return converted

    # ------------------------------------------------------------------
    # Multimodal file extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_file_refs_from_content(
        content: list, msg_index: int
    ) -> Tuple[List[str], List[FileRef]]:
        """Separate text parts and file references from multimodal content blocks.

        Supported block types (LangGraph SDK / agent-chat-ui format):

        - ``image``: ``{"type": "image", "mimeType": "image/png", "data": "<b64>"}``
        - ``file``: ``{"type": "file", "mimeType": "application/pdf", "data": "<b64>"}``
        - ``tess_ai``: ``{"type": "tess_ai", "file_id": 55}`` (direct Tess reference)
        - ``url``: ``{"type": "url", "url": "https://...", "mimeType": "text/csv"}``
        """
        text_parts: List[str] = []
        file_refs: List[FileRef] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))

            elif block_type in ("image", "file"):
                raw_data = block.get("data")
                mime = block.get("mimeType", "application/octet-stream")
                if raw_data:
                    file_refs.append(FileRef(
                        data=base64.b64decode(raw_data),
                        mime_type=mime,
                        message_index=msg_index,
                    ))

            elif block_type == "tess_ai":
                file_refs.append(FileRef(
                    file_id=block["file_id"],
                    message_index=msg_index,
                ))

            elif block_type == "url":
                file_refs.append(FileRef(
                    url=block["url"],
                    mime_type=block.get("mimeType", ""),
                    message_index=msg_index,
                ))

        return text_parts, file_refs

    @staticmethod
    def _convert_messages_with_files(
        messages: List[BaseMessage],
    ) -> Tuple[List[dict], List[FileRef]]:
        """Convert LangChain messages to Tess format, extracting file references.

        Like :meth:`_convert_messages` but also inspects multimodal
        ``HumanMessage`` content blocks for ``document`` entries.  Returns
        plain-text Tess messages **and** a list of :class:`FileRef` objects
        that must be resolved into ``file_ids`` before execution.
        """
        converted: List[dict] = []
        all_file_refs: List[FileRef] = []

        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                text_parts, file_refs = ChatTessAI._extract_file_refs_from_content(
                    msg.content, idx
                )
                all_file_refs.extend(file_refs)
                role = "user"
                content = "\n".join(text_parts).strip() or "[See attached files]"
            elif isinstance(msg, SystemMessage):
                role, content = "developer", ChatTessAI._content_to_str(msg.content)
            elif isinstance(msg, HumanMessage):
                role, content = "user", ChatTessAI._content_to_str(msg.content)
            elif isinstance(msg, AIMessage):
                role, content = "assistant", ChatTessAI._assistant_message_to_tess_content(msg)
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None) or "unknown_tool"
                role = "user"
                content = (
                    f'[Command Result] The command "{tool_name}" returned:\n'
                    f"{ChatTessAI._content_to_str(msg.content)}\n\n"
                    f'[Respond with a JSON object: {{"content": "your response"}}]'
                )
            else:
                role, content = "user", ChatTessAI._content_to_str(msg.content)

            if converted and converted[-1]["role"] == role and role != "developer":
                converted[-1]["content"] += f"\n\n{content}"
            else:
                converted.append({"role": role, "content": content})

        return converted, all_file_refs

    # ------------------------------------------------------------------
    # Payload / request helpers
    # ------------------------------------------------------------------

    @property
    def _execute_url(self) -> str:
        return f"{self.base_url}/agents/{self.agent_id}/execute"

    def _response_url(self, response_id: int) -> str:
        return f"{self.base_url}/agent-responses/{response_id}"

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "x-workspace-id": str(self.workspace_id),
        }

    def _build_payload(
        self,
        messages: List[BaseMessage],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        return self._build_payload_from_converted(
            self._convert_messages(messages), stream=stream, **kwargs
        )

    def _build_payload_from_converted(
        self,
        converted_messages: List[dict],
        *,
        stream: bool = False,
        root_id: Optional[int] = None,
        file_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": f"{self.temperature:g}",
            "tools": self.tools,
            "messages": converted_messages,
            "wait_execution": self.wait_execution if not stream else False,
        }
        if root_id is not None:
            payload["root_id"] = root_id
        if stream:
            payload["stream"] = True
        effective_file_ids = file_ids if file_ids is not None else self.file_ids
        if effective_file_ids:
            payload["file_ids"] = effective_file_ids
        payload.update(kwargs)
        return payload

    # ------------------------------------------------------------------
    # HTTP clients
    # ------------------------------------------------------------------

    def _sync_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout)

    @property
    def _upload_headers(self) -> dict:
        """Headers for file upload requests (no Content-Type -- httpx sets multipart)."""
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "x-workspace-id": str(self.workspace_id),
        }

    # ------------------------------------------------------------------
    # File upload + processing
    # ------------------------------------------------------------------

    def _upload_and_process_file(
        self,
        client: httpx.Client,
        file_bytes: bytes,
        filename: str,
    ) -> int:
        """Upload a file to Tess with ``process=true`` and poll until done.

        Returns the Tess ``file_id``.  Raises :class:`RuntimeError` if
        processing fails.
        """
        resp = client.post(
            f"{self.base_url}/files",
            headers=self._upload_headers,
            files={"file": (filename, file_bytes)},
            data={"process": "true"},
        )
        resp.raise_for_status()
        file_data = resp.json()

        file_id: int = file_data["id"]
        status = file_data.get("status", "")

        if status in _FILE_TERMINAL_STATUSES:
            if status == "failed":
                raise RuntimeError(
                    f"Tess file processing failed for file_id={file_id}: {file_data}"
                )
            return file_id

        poll_url = f"{self.base_url}/files/{file_id}"
        while True:
            time.sleep(self.polling_interval)
            poll_resp = client.get(poll_url, headers=self._upload_headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            poll_status = poll_data.get("status", "")

            if poll_status == "completed":
                return file_id
            if poll_status == "failed":
                raise RuntimeError(
                    f"Tess file processing failed for file_id={file_id}: {poll_data}"
                )

    async def _aupload_and_process_file(
        self,
        client: httpx.AsyncClient,
        file_bytes: bytes,
        filename: str,
    ) -> int:
        """Async version of :meth:`_upload_and_process_file`."""
        import asyncio

        resp = await client.post(
            f"{self.base_url}/files",
            headers=self._upload_headers,
            files={"file": (filename, file_bytes)},
            data={"process": "true"},
        )
        resp.raise_for_status()
        file_data = resp.json()

        file_id: int = file_data["id"]
        status = file_data.get("status", "")

        if status in _FILE_TERMINAL_STATUSES:
            if status == "failed":
                raise RuntimeError(
                    f"Tess file processing failed for file_id={file_id}: {file_data}"
                )
            return file_id

        poll_url = f"{self.base_url}/files/{file_id}"
        while True:
            await asyncio.sleep(self.polling_interval)
            poll_resp = await client.get(poll_url, headers=self._upload_headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            poll_status = poll_data.get("status", "")

            if poll_status == "completed":
                return file_id
            if poll_status == "failed":
                raise RuntimeError(
                    f"Tess file processing failed for file_id={file_id}: {poll_data}"
                )

    # ------------------------------------------------------------------
    # File ID resolution
    # ------------------------------------------------------------------

    def _resolve_file_ids(
        self,
        client: httpx.Client,
        file_refs: List[FileRef],
    ) -> List[int]:
        """Resolve a list of :class:`FileRef` into Tess ``file_id`` integers.

        Uses an in-memory hash→file_id cache to avoid redundant uploads
        within the same session.  On cold start the Tess server-side
        deduplication guarantees the same ``file_id`` for identical content.
        """
        file_ids: List[int] = []

        for ref in file_refs:
            if ref.file_id is not None:
                file_ids.append(ref.file_id)
                continue

            if ref.url and ref.data is None:
                dl_resp = client.get(ref.url)
                dl_resp.raise_for_status()
                ref.data = dl_resp.content
                if not ref.mime_type:
                    ct = dl_resp.headers.get("content-type", "application/octet-stream")
                    ref.mime_type = ct.split(";")[0].strip()

            content_hash = ref.content_hash
            if content_hash is not None:
                with self._lock:
                    cached = self._file_cache.get(content_hash)
                if cached is not None:
                    file_ids.append(cached)
                    continue

            fid = self._upload_and_process_file(
                client, ref.data, ref.upload_filename  # type: ignore[arg-type]
            )

            if content_hash is not None:
                with self._lock:
                    self._file_cache[content_hash] = fid

            file_ids.append(fid)

        return file_ids

    async def _aresolve_file_ids(
        self,
        client: httpx.AsyncClient,
        file_refs: List[FileRef],
    ) -> List[int]:
        """Async version of :meth:`_resolve_file_ids`."""
        file_ids: List[int] = []

        for ref in file_refs:
            if ref.file_id is not None:
                file_ids.append(ref.file_id)
                continue

            if ref.url and ref.data is None:
                dl_resp = await client.get(ref.url)
                dl_resp.raise_for_status()
                ref.data = dl_resp.content
                if not ref.mime_type:
                    ct = dl_resp.headers.get("content-type", "application/octet-stream")
                    ref.mime_type = ct.split(";")[0].strip()

            content_hash = ref.content_hash
            if content_hash is not None:
                with self._lock:
                    cached = self._file_cache.get(content_hash)
                if cached is not None:
                    file_ids.append(cached)
                    continue

            fid = await self._aupload_and_process_file(
                client, ref.data, ref.upload_filename  # type: ignore[arg-type]
            )

            if content_hash is not None:
                with self._lock:
                    self._file_cache[content_hash] = fid

            file_ids.append(fid)

        return file_ids

    @staticmethod
    def _merge_file_ids(
        static_ids: Optional[List[int]],
        dynamic_ids: List[int],
    ) -> Optional[List[int]]:
        """Merge static (constructor) and dynamic (per-request) file IDs, deduped."""
        combined = list(dict.fromkeys((static_ids or []) + dynamic_ids))
        return combined or None

    # ------------------------------------------------------------------
    # Polling (for wait_execution=False)
    # ------------------------------------------------------------------

    _TERMINAL_STATUSES = {"succeeded", "failed", "error", "completed"}

    @staticmethod
    def _is_completed(data: dict) -> bool:
        """Check if an execute response already contains the final output."""
        responses = data.get("responses", [])
        if responses:
            status = responses[0].get("status", "")
            return status in ("succeeded", "completed")
        return False

    @staticmethod
    def _extract_response_id(data: dict) -> int:
        responses = data.get("responses", [])
        if responses:
            return responses[0]["id"]
        raise ValueError(f"No response id found in API response: {data}")

    def _poll_for_result(self, client: httpx.Client, response_id: int) -> dict:
        url = self._response_url(response_id)
        while True:
            resp = client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in self._TERMINAL_STATUSES:
                if data.get("status") in ("failed", "error"):
                    raise RuntimeError(
                        f"Tess AI execution failed: "
                        f"{data.get('error', 'unknown error')}"
                    )
                return data
            time.sleep(self.polling_interval)

    async def _apoll_for_result(
        self, client: httpx.AsyncClient, response_id: int
    ) -> dict:
        import asyncio

        url = self._response_url(response_id)
        while True:
            resp = await client.get(url, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in self._TERMINAL_STATUSES:
                if data.get("status") in ("failed", "error"):
                    raise RuntimeError(
                        f"Tess AI execution failed: "
                        f"{data.get('error', 'unknown error')}"
                    )
                return data
            await asyncio.sleep(self.polling_interval)

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    def _extract_output_and_metadata(self, data: dict) -> tuple[str, dict]:
        """Extract output text and metadata from a POST /execute response."""
        responses = data.get("responses", [])
        if not responses:
            raise ValueError(f"Empty responses from Tess API: {data}")
        entry = responses[0]
        return entry.get("output", ""), {
            "model_name": self.model,
            "tess_response_id": entry.get("id"),
            "tess_root_id": entry.get("root_id"),
            "tess_status": entry.get("status"),
            "credits": entry.get("credits", 0),
            "template_id": entry.get("template_id"),
        }

    def _extract_output_and_metadata_from_poll(self, data: dict) -> tuple[str, dict]:
        """Extract output text and metadata from a GET /agent-responses response."""
        return data.get("output", ""), {
            "model_name": self.model,
            "tess_response_id": data.get("id"),
            "tess_root_id": data.get("root_id"),
            "tess_status": data.get("status"),
            "credits": data.get("credits", 0),
            "template_id": data.get("template_id"),
        }

    # ------------------------------------------------------------------
    # Retry error raising
    # ------------------------------------------------------------------

    def _raise_after_retries(
        self,
        last_error: Optional[Exception],
        last_output: str,
    ) -> None:
        """Raise an appropriate exception after all retries are exhausted."""
        if isinstance(last_error, TessAPIError):
            raise last_error
        if isinstance(last_error, (httpx.HTTPStatusError, httpx.TransportError)):
            raise last_error
        if isinstance(last_error, _EmptyResponseError):
            raise ValueError(
                f"Tess API returned an empty response after "
                f"{self.max_retries + 1} attempts"
            )
        if isinstance(last_error, ToolCallParseError):
            raise ValueError(
                f"Failed to parse tool calls from model response after "
                f"{self.max_retries + 1} attempts: {last_output}"
            )
        raise ValueError(
            f"Tess API request failed after {self.max_retries + 1} attempts"
        )

    # ------------------------------------------------------------------
    # SSE streaming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sse_line(line: str) -> Optional[dict]:
        """Extract JSON data from an SSE 'data:' line."""
        line = line.strip()
        if not line or not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _iter_sse_chunks(
        self,
        lines: Iterator[str],
        run_manager: Optional[CallbackManagerForLLMRun],
    ) -> Iterator[ChatGenerationChunk]:
        for line in lines:
            event = self._parse_sse_line(line)
            if event is None:
                continue

            status = event.get("status", "")
            output = event.get("output", "")

            if output:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=output)
                )
                if run_manager:
                    run_manager.on_llm_new_token(output, chunk=chunk)
                yield chunk

            if status in self._TERMINAL_STATUSES:
                final_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={
                            "tess_response_id": event.get("id"),
                            "tess_root_id": event.get("root_id"),
                            "tess_status": status,
                            "credits": event.get("credits"),
                            "template_id": event.get("template_id"),
                        },
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=final_chunk)
                yield final_chunk
                return

    async def _aiter_sse_chunks(
        self,
        lines: AsyncIterator[str],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for line in lines:
            event = self._parse_sse_line(line)
            if event is None:
                continue

            status = event.get("status", "")
            output = event.get("output", "")

            if output:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=output)
                )
                if run_manager:
                    await run_manager.on_llm_new_token(output, chunk=chunk)
                yield chunk

            if status in self._TERMINAL_STATUSES:
                final_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={
                            "tess_response_id": event.get("id"),
                            "tess_root_id": event.get("root_id"),
                            "tess_status": status,
                            "credits": event.get("credits"),
                            "template_id": event.get("template_id"),
                        },
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token("", chunk=final_chunk)
                yield final_chunk
                return
