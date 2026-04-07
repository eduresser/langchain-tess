# @langchain/tess

LangChain.js integration for the [Tess AI](https://tess.im) API.

## Installation

```bash
npm install @langchain/tess @langchain/core
```

## Quick Start

```typescript
import { ChatTessAI } from "@langchain/tess";

const llm = new ChatTessAI({
  apiKey: "YOUR_TESS_API_KEY", // or set TESS_API_KEY env var
  agentId: 8794,
  workspaceId: 1,
  model: "tess-5",
  temperature: 0.5,
});

// Simple invoke
const response = await llm.invoke("Hello, how can you help me?");
console.log(response.content);

// Streaming
for await (const chunk of await llm.stream("Tell me a short story")) {
  process.stdout.write(String(chunk.content));
}

// Batch
const responses = await llm.batch(["Hello", "Goodbye"]);
```

## Message History

```typescript
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const response = await llm.invoke([
  new SystemMessage("You are a helpful assistant."),
  new HumanMessage("Hello!"),
  new AIMessage("Hi! How can I help?"),
  new HumanMessage("What is the capital of Brazil?"),
]);
```

## Tool Calling (Prompt-Based)

The Tess AI API does not support native tool calling. This integration implements
tool calling via a JSON protocol injected into the system prompt. The model is
instructed to respond with structured JSON containing tool calls.

```typescript
import { ChatTessAI } from "@langchain/tess";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const getWeather = tool(
  async ({ city }) => `Sunny in ${city}`,
  {
    name: "get_weather",
    description: "Get the weather for a city",
    schema: z.object({ city: z.string() }),
  }
);

const llmWithTools = llm.bindTools([getWeather]);
const response = await llmWithTools.invoke("What's the weather in SP?");
console.log(response.tool_calls);
```

## Structured Output

```typescript
import { z } from "zod";

const schema = z.object({
  answer: z.string(),
  confidence: z.number(),
});

const structured = llm.withStructuredOutput({
  title: "Answer",
  description: "A structured answer",
  properties: {
    answer: { type: "string" },
    confidence: { type: "number" },
  },
  required: ["answer", "confidence"],
});

const result = await structured.invoke("What is 2+2?");
console.log(result); // { answer: "4", confidence: 1.0 }
```

## Configuration

| Parameter              | Type       | Default                 | Description                                                 |
| ---------------------- | ---------- | ----------------------- | ----------------------------------------------------------- |
| `apiKey`               | `string`   | env `TESS_API_KEY`      | Tess AI API key                                             |
| `agentId`              | `number`   | -                       | The Tess AI agent ID                                        |
| `workspaceId`          | `number`   | -                       | Tess workspace ID                                           |
| `model`                | `string`   | `"tess-5"`              | Model to use                                                |
| `temperature`          | `number`   | `1.0`                   | Temperature for generation (0.0 - 1.0)                      |
| `tools`                | `string`   | `"no-tools"`            | Tess tools (`"internet"`, `"twitter"`, `"wikipedia"`, etc.) |
| `baseUrl`              | `string`   | `"https://api.tess.im"` | API base URL                                                |
| `timeout`              | `number`   | `120`                   | Request timeout in seconds                                  |
| `maxRetries`           | `number`   | `5`                     | Max retry attempts                                          |
| `waitExecution`        | `boolean`  | `true`                  | Wait for execution to complete (100s API timeout)           |
| `pollingInterval`      | `number`   | `5.0`                   | Polling interval in seconds                                 |
| `fileIds`              | `number[]` | -                       | File IDs to attach to every execution                       |
| `trackConversations`   | `boolean`  | `true`                  | Reuse root_id for conversation continuations                |
| `maxTrackedConversations` | `number` | `100`                  | Max conversations in the LRU cache                          |

## File Uploads (Multimodal)

Send files (images, PDFs, CSVs, etc.) using multimodal content blocks.
The provider automatically uploads files to Tess, waits for processing, and
attaches the resulting `file_ids` to each execution.

```typescript
import { HumanMessage } from "@langchain/core/messages";
import { readFileSync } from "fs";

const pdfB64 = readFileSync("report.pdf").toString("base64");

// PDF / CSV / any file
const response = await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Summarize this report" },
      { type: "file", mimeType: "application/pdf", data: pdfB64 },
    ],
  }),
]);

// Image
const response2 = await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Describe this image" },
      { type: "image", mimeType: "image/png", data: imgB64 },
    ],
  }),
]);

// External URL
const response3 = await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Analyze this dataset" },
      { type: "url", url: "https://example.com/data.csv", mimeType: "text/csv" },
    ],
  }),
]);

// Direct Tess file_id
const response4 = await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "What does this image show?" },
      { type: "tess_ai", file_id: 73325 },
    ],
  }),
]);
```

## Conversation Tracking

When `trackConversations` is enabled (default), the integration automatically
tracks conversation history and sends only new (delta) messages on subsequent
calls with the same message prefix. This reduces API payload size and leverages
Tess AI's server-side conversation continuations via `root_id`.

```typescript
// First call - sends full history
const r1 = await llm.invoke([new HumanMessage("Hello")]);

// Second call - sends only the delta (new messages)
const r2 = await llm.invoke([
  new HumanMessage("Hello"),
  new AIMessage(r1.content as string),
  new HumanMessage("Tell me more"),
]);

// Reset cache
llm.resetConversations();
```

## Available Models

Some of the models available through Tess AI:

- `tess-5`, `tess-6`, `tess-6.1`
- `gpt-4o`, `gpt-4o-mini`, `gpt-5`
- `claude-4-sonnet`, `claude-4-opus`
- `gemini-2.5-pro`, `gemini-2.5-flash`
- And many more - check the [Tess AI docs](https://docs.tess.im/en/models-and-cost) for the full list.

## API Documentation

- [Tess AI API Overview](https://docs.tess.im/en/api-overview)
- [Execute Agent](https://docs.tess.im/en/execute-agent)

## License

MIT
