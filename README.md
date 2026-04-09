# langchain-tessai

LangChain integration for the [Tess AI](https://tess.im) API.

## Installation

```bash
pip install langchain-tessai
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from langchain_tessai import ChatTessAI

llm = ChatTessAI(
    api_key="YOUR_TESSAI_API_KEY",
    agent_id=8794,
    model="tess-5",
    temperature=0.5,
)

# Simple invoke
response = llm.invoke("Hello, how can you help me?")
print(response.content)

# Streaming
for chunk in llm.stream("Tell me a short story"):
    print(chunk.content, end="")

# Batch
responses = llm.batch(["Hello", "Goodbye"])

# With Tess tools (internet search, etc.)
llm_web = ChatTessAI(
    api_key="YOUR_TESSAI_API_KEY",
    agent_id=8794,
    model="auto",
    tools="search_engines",
)
response = llm_web.invoke("What are the latest news?")
```

## Message History

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I help?"),
    HumanMessage(content="What is the capital of Brazil?"),
])
```

## Configuration


| Parameter          | Type             | Default                 | Description                                                 |
| ------------------ | ---------------- | ----------------------- | ----------------------------------------------------------- |
| `api_key`          | `str`            | -                       | Tess AI API key (or set `TESSAI_API_KEY` env var)           |
| `agent_id`         | `int`            | -                       | The Tess AI agent ID                                        |
| `model`            | `str`            | `"tess-5"`              | Model to use                                                |
| `temperature`      | `float`          | `1.0`                   | Temperature for generation                                  |
| `tools`            | `str`            | `"no-tools"`            | Tess tools (`"internet"`, `"twitter"`, `"wikipedia"`, etc.) |
| `workspace_id`     | `int|None`       | `None`                  | Workspace ID                                                |
| `base_url`         | `str`            | `"https://api.tess.im"` | API base URL                                                |
| `timeout`          | `int`            | `120`                   | Request timeout in seconds                                  |
| `max_retries`      | `int`            | `2`                     | Max retry attempts                                          |
| `wait_execution`   | `bool`           | `True`                  | Wait for execution to complete (100s API timeout)           |
| `polling_interval` | `float`          | `2.0`                   | Polling interval in seconds when `wait_execution=False`     |
| `file_ids`         | `list[int]|None` | `None`                  | File IDs to attach to execution                             |


## File Uploads (Multimodal)

Send files (images, PDFs, CSVs, etc.) using LangGraph SDK content blocks.
The provider automatically uploads files to Tess, waits for processing, and
attaches the resulting `file_ids` to each execution. Compatible with the
`agent-chat-ui` file upload out of the box.

```python
import base64

with open("report.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

# PDF / CSV / any file
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Summarize this report"},
    {"type": "file", "mimeType": "application/pdf", "data": pdf_b64},
])])

# Image
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "mimeType": "image/png", "data": img_b64},
])])

# External URL
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "Analyze this dataset"},
    {"type": "url", "url": "https://example.com/data.csv", "mimeType": "text/csv"},
])])

# Direct Tess file_id (if you already uploaded via the API)
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "What does this image show?"},
    {"type": "tess_ai", "file_id": 73325},
])])
```

Files are deduplicated server-side by Tess — re-uploading the same file returns the
original `file_id`. An in-memory cache avoids redundant uploads within the same session.

## Available Models

Some of the models available through Tess AI:

- `tess-5`, `tess-6`, `tess-6.1`
- `gpt-4o`, `gpt-4o-mini`, `gpt-5`
- `claude-4-sonnet`, `claude-4-opus`
- `gemini-2.5-pro`, `gemini-2.5-flash`
- And many more — check the [Tess AI docs](https://docs.tess.im/en/models-and-cost) for the full list.

## API Documentation

- [Tess AI API Overview](https://docs.tess.im/en/api-overview)
- [Execute Agent](https://docs.tess.im/en/execute-agent)
- [Stream Agent](https://docs.tess.im/en/stream-agent)

