import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatTessAI } from "../chat_models.js";
import {
  TessAuthenticationError,
  TessPayloadTooLargeError,
  TessRateLimitError,
  TessServerError,
} from "../exceptions.js";

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function makeLlm(overrides: Record<string, unknown> = {}): ChatTessAI {
  return new ChatTessAI({
    apiKey: "test-key",
    agentId: 8794,
    workspaceId: 1,
    model: "tess-5",
    temperature: 0.5,
    maxRetries: 0,
    ...overrides,
  });
}

function mockFetchSuccess(
  output: string,
  meta?: Record<string, unknown>,
) {
  const response = {
    responses: [
      {
        id: 100,
        status: "succeeded",
        output,
        root_id: 5001,
        credits: 0.01,
        template_id: 8794,
        ...meta,
      },
    ],
  };
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve(response),
  });
}

function mockFetchError(statusCode: number, body: Record<string, unknown>) {
  return vi.fn().mockResolvedValue({
    ok: false,
    status: statusCode,
    json: () => Promise.resolve(body),
  });
}

// ------------------------------------------------------------------
// Properties
// ------------------------------------------------------------------

describe("ChatTessAI properties", () => {
  it("has correct _llmType", () => {
    const llm = makeLlm();
    expect(llm._llmType()).toBe("tess-ai");
  });

  it("has correct _identifyingParams", () => {
    const llm = makeLlm();
    expect(llm._identifyingParams()).toEqual({
      model_name: "tess-5",
      agent_id: 8794,
      temperature: 0.5,
      tools: "no-tools",
    });
  });

  it("uses default values", () => {
    const llm = makeLlm();
    expect(llm.baseUrl).toBe("https://api.tess.im");
    expect(llm.timeout).toBe(120);
    expect(llm.waitExecution).toBe(true);
    expect(llm.trackConversations).toBe(true);
    expect(llm.maxTrackedConversations).toBe(100);
  });

  it("reads api key from constructor", () => {
    const llm = makeLlm({ apiKey: "my-key" });
    expect(llm.apiKey).toBe("my-key");
  });

  it("static lc_name returns ChatTessAI", () => {
    expect(ChatTessAI.lc_name()).toBe("ChatTessAI");
  });
});

// ------------------------------------------------------------------
// Message conversion (access via _generate side-effects)
// ------------------------------------------------------------------

describe("Message conversion", () => {
  let llm: ChatTessAI;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    llm = makeLlm();
    fetchMock = mockFetchSuccess('{"content": "Hello!"}');
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("converts basic messages and calls API", async () => {
    await llm.invoke([new HumanMessage("Hi")]);
    expect(fetchMock).toHaveBeenCalled();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://api.tess.im/agents/8794/execute");
    const body = JSON.parse(init.body);
    expect(body.model).toBe("tess-5");
    expect(body.temperature).toBe("0.5");
    expect(body.messages).toBeInstanceOf(Array);
    expect(body.messages.length).toBeGreaterThanOrEqual(2);
    // First message should be developer (system with JSON prompt)
    expect(body.messages[0].role).toBe("developer");
    // Last message should be user
    expect(body.messages[body.messages.length - 1].role).toBe("user");
    expect(body.messages[body.messages.length - 1].content).toBe("Hi");
  });

  it("includes authorization header", async () => {
    await llm.invoke("Hello");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer test-key");
    expect(init.headers["x-workspace-id"]).toBe("1");
  });

  it("merges consecutive user messages", async () => {
    await llm.invoke([new HumanMessage("msg1"), new HumanMessage("msg2")]);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    const userMsgs = body.messages.filter(
      (m: Record<string, string>) => m.role === "user",
    );
    expect(userMsgs).toHaveLength(1);
    expect(userMsgs[0].content).toContain("msg1");
    expect(userMsgs[0].content).toContain("msg2");
  });

  it("converts SystemMessage to developer role", async () => {
    await llm.invoke([
      new SystemMessage("Be helpful"),
      new HumanMessage("Hi"),
    ]);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.messages[0].role).toBe("developer");
    expect(body.messages[0].content).toContain("Be helpful");
  });

  it("converts ToolMessage to user role", async () => {
    await llm.invoke([
      new HumanMessage("What's the weather?"),
      new AIMessage({
        content: "calling tool",
        tool_calls: [
          { name: "get_weather", args: { city: "SP" }, id: "call_1" },
        ],
      }),
      new ToolMessage({
        content: "Sunny 25°C",
        tool_call_id: "call_1",
        name: "get_weather",
      }),
    ]);
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    const lastMsg = body.messages[body.messages.length - 1];
    expect(lastMsg.role).toBe("user");
    expect(lastMsg.content).toContain("[Tool Result]");
    expect(lastMsg.content).toContain("get_weather");
    expect(lastMsg.content).toContain("Sunny 25°C");
  });
});

// ------------------------------------------------------------------
// _generate with successful responses
// ------------------------------------------------------------------

describe("_generate success", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns content from JSON response", async () => {
    const llm = makeLlm();
    vi.stubGlobal("fetch", mockFetchSuccess('{"content": "Hello there!"}'));
    const result = await llm.invoke("Hi");
    expect(result.content).toBe("Hello there!");
  });

  it("includes response metadata", async () => {
    const llm = makeLlm();
    vi.stubGlobal("fetch", mockFetchSuccess('{"content": "Hi"}'));
    const result = await llm.invoke("Hello");
    expect(result.response_metadata).toBeDefined();
    expect(result.response_metadata.tess_response_id).toBe(100);
    expect(result.response_metadata.tess_root_id).toBe(5001);
    expect(result.response_metadata.model_name).toBe("tess-5");
  });

  it("includes usage metadata", async () => {
    const llm = makeLlm();
    vi.stubGlobal("fetch", mockFetchSuccess('{"content": "Hi"}'));
    const result = await llm.invoke("Hello");
    expect(result.usage_metadata).toBeDefined();
    expect(result.usage_metadata!.input_tokens).toBeGreaterThan(0);
    expect(result.usage_metadata!.output_tokens).toBeGreaterThan(0);
    expect(result.usage_metadata!.total_tokens).toBe(
      result.usage_metadata!.input_tokens +
        result.usage_metadata!.output_tokens,
    );
  });

  it("parses tool calls from response", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess(
        '{"content": "calling", "tool_calls": [{"name": "get_weather", "arguments": {"city": "SP"}}]}',
      ),
    );
    const result = (await llm.invoke("weather?")) as AIMessage;
    expect(result.tool_calls).toHaveLength(1);
    expect(result.tool_calls![0].name).toBe("get_weather");
    expect(result.tool_calls![0].args).toEqual({ city: "SP" });
  });
});

// ------------------------------------------------------------------
// Stop sequences
// ------------------------------------------------------------------

describe("Stop sequences", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("truncates at first stop sequence", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "Hello STOP World"}'),
    );
    const result = await llm.invoke("Hi", { stop: ["STOP"] });
    expect(result.content).toBe("Hello ");
  });

  it("returns full content when no stop sequence found", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "Hello World"}'),
    );
    const result = await llm.invoke("Hi", { stop: ["XXX"] });
    expect(result.content).toBe("Hello World");
  });
});

// ------------------------------------------------------------------
// Error handling
// ------------------------------------------------------------------

describe("Error handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("throws TessAuthenticationError on 403 without retry", async () => {
    const llm = makeLlm({ maxRetries: 3 });
    vi.stubGlobal(
      "fetch",
      mockFetchError(403, { error: "invalid key" }),
    );
    await expect(llm.invoke("Hi")).rejects.toThrow(
      TessAuthenticationError,
    );
  });

  it("throws TessPayloadTooLargeError on 413 without retry", async () => {
    const llm = makeLlm({ maxRetries: 3 });
    vi.stubGlobal(
      "fetch",
      mockFetchError(413, { error: "too large" }),
    );
    await expect(llm.invoke("Hi")).rejects.toThrow(
      TessPayloadTooLargeError,
    );
  });

  it("retries on 500 error", async () => {
    const llm = makeLlm({ maxRetries: 1 });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: "server error" }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: '{"content": "recovered"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);
    const result = await llm.invoke("Hi");
    expect(result.content).toBe("recovered");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("retries on rate limit with retryAfter", async () => {
    const llm = makeLlm({ maxRetries: 1 });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () =>
          Promise.resolve({
            error: "rate limited",
            retry_after: 0.01,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: '{"content": "ok"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);
    const result = await llm.invoke("Hi");
    expect(result.content).toBe("ok");
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("retries on empty response", async () => {
    const llm = makeLlm({ maxRetries: 1 });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: "   ",
                root_id: 5001,
                credits: 0,
                template_id: 8794,
              },
            ],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 101,
                status: "succeeded",
                output: '{"content": "real response"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);
    const result = await llm.invoke("Hi");
    expect(result.content).toBe("real response");
  });

  it("fails after max retries exhausted", async () => {
    const llm = makeLlm({ maxRetries: 1 });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: "server error" }),
      }),
    );
    await expect(llm.invoke("Hi")).rejects.toThrow();
  });
});

// ------------------------------------------------------------------
// Polling
// ------------------------------------------------------------------

describe("Polling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("polls when initial response is not completed", async () => {
    const llm = makeLlm({ pollingInterval: 0.01 });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              { id: 100, status: "starting", output: "", root_id: 100 },
            ],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            id: 100,
            status: "succeeded",
            output: '{"content": "polled result"}',
            root_id: 5001,
            credits: 0.01,
            template_id: 8794,
          }),
      });
    vi.stubGlobal("fetch", fetchFn);
    const result = await llm.invoke("Hi");
    expect(result.content).toBe("polled result");
    expect(fetchFn).toHaveBeenCalledTimes(2);
    // Second call should be GET to agent-responses
    const [pollUrl] = fetchFn.mock.calls[1];
    expect(pollUrl).toContain("/agent-responses/100");
  });
});

// ------------------------------------------------------------------
// Conversation tracking
// ------------------------------------------------------------------

describe("Conversation tracking", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends delta messages on second call", async () => {
    const llm = makeLlm({ trackConversations: true });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: '{"content": "first response"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 101,
                status: "succeeded",
                output: '{"content": "second response"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);

    // First call - full conversation
    await llm.invoke([new HumanMessage("Hello")]);
    const body1 = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body1.root_id).toBeUndefined();

    // Second call - should find conversation and send delta
    await llm.invoke([
      new HumanMessage("Hello"),
      new AIMessage({ content: "first response" }),
      new HumanMessage("Follow up"),
    ]);
    const body2 = JSON.parse(fetchFn.mock.calls[1][1].body);
    expect(body2.root_id).toBe(5001);
    // Delta should be shorter than full conversation
    const fullMsgCount = 4; // developer + user + assistant + user (with JSON prompt)
    expect(body2.messages.length).toBeLessThan(fullMsgCount);
  });

  it("sends full messages when tracking disabled", async () => {
    const llm = makeLlm({ trackConversations: false });
    const fetchFn = vi
      .fn()
      .mockResolvedValue({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: '{"content": "response"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);

    await llm.invoke([new HumanMessage("Hello")]);
    await llm.invoke([
      new HumanMessage("Hello"),
      new AIMessage({ content: "response" }),
      new HumanMessage("Follow up"),
    ]);

    const body2 = JSON.parse(fetchFn.mock.calls[1][1].body);
    expect(body2.root_id).toBeUndefined();
  });

  it("resetConversations clears cache", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "response"}'),
    );

    await llm.invoke([new HumanMessage("Hello")]);
    llm.resetConversations();

    await llm.invoke([
      new HumanMessage("Hello"),
      new AIMessage({ content: "response" }),
      new HumanMessage("Follow up"),
    ]);

    const body = JSON.parse(
      (vi.mocked(fetch).mock.calls[1][1] as RequestInit).body as string,
    );
    expect(body.root_id).toBeUndefined();
  });
});

// ------------------------------------------------------------------
// Streaming (bypass)
// ------------------------------------------------------------------

describe("Streaming (sync bypass)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("yields single chunk with full response", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "streamed response"}'),
    );

    const chunks = [];
    for await (const chunk of llm._streamResponseChunks(
      [new HumanMessage("Hi")],
      {} as ChatTessAI["ParsedCallOptions"],
    )) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].message.content).toBe("streamed response");
  });
});

// ------------------------------------------------------------------
// Build payload
// ------------------------------------------------------------------

describe("Build payload", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("includes file_ids when set", async () => {
    const llm = makeLlm({ fileIds: [55, 72] });
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "response"}'),
    );
    await llm.invoke("Hi");
    const body = JSON.parse(
      (vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.file_ids).toEqual([55, 72]);
  });

  it("uses wait_execution from config", async () => {
    const llm = makeLlm({ waitExecution: false, pollingInterval: 0.01 });
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              { id: 100, status: "starting", output: "", root_id: 100 },
            ],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            id: 100,
            status: "succeeded",
            output: '{"content": "done"}',
            root_id: 5001,
            credits: 0.01,
            template_id: 8794,
          }),
      });
    vi.stubGlobal("fetch", fetchFn);
    await llm.invoke("Hi");
    const body = JSON.parse(fetchFn.mock.calls[0][1].body);
    expect(body.wait_execution).toBe(false);
  });

  it("includes tools in payload from bind", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess(
        '{"content": "result", "tool_calls": [{"name": "get_weather", "arguments": {"city": "SP"}}]}',
      ),
    );

    const toolDef = {
      type: "function",
      function: {
        name: "get_weather",
        description: "Get weather",
        parameters: {
          type: "object",
          properties: {
            city: { type: "string", description: "City name" },
          },
          required: ["city"],
        },
      },
    };

    const llmWithTools = llm.bind({
      tools: [toolDef],
    } as Partial<ChatTessAICallOptions>);

    const result = (await llmWithTools.invoke("weather?")) as AIMessage;
    expect(result.tool_calls).toHaveLength(1);
    // The JSON prompt should include tool definitions
    const body = JSON.parse(
      (vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.messages[0].content).toContain("get_weather");
  });
});

// ------------------------------------------------------------------
// Token counting
// ------------------------------------------------------------------

describe("Token counting", () => {
  it("estimates token count", async () => {
    const llm = makeLlm();
    const count = await llm.getNumTokens("Hello world");
    expect(count).toBeGreaterThan(0);
    expect(typeof count).toBe("number");
  });
});

// ------------------------------------------------------------------
// Multimodal files
// ------------------------------------------------------------------

describe("Multimodal file handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("extracts tess_ai file refs without upload", async () => {
    const llm = makeLlm();
    vi.stubGlobal(
      "fetch",
      mockFetchSuccess('{"content": "analyzed"}'),
    );

    await llm.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "Analyze this" },
          { type: "tess_ai", file_id: 123 },
        ],
      }),
    ]);

    const body = JSON.parse(
      (vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.file_ids).toContain(123);
  });

  it("uploads base64 files", async () => {
    const llm = makeLlm({ pollingInterval: 0.01 });
    const b64Data = Buffer.from("test content").toString("base64");

    const fetchFn = vi
      .fn()
      // File upload response
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({ id: 555, status: "completed" }),
      })
      // Execute response
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            responses: [
              {
                id: 100,
                status: "succeeded",
                output: '{"content": "processed"}',
                root_id: 5001,
                credits: 0.01,
                template_id: 8794,
              },
            ],
          }),
      });
    vi.stubGlobal("fetch", fetchFn);

    await llm.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "Process this" },
          {
            type: "file",
            mimeType: "application/pdf",
            data: b64Data,
          },
        ],
      }),
    ]);

    // First call should be file upload
    expect(fetchFn.mock.calls[0][0]).toContain("/files");
    // Second call should be execute with file_ids
    const execBody = JSON.parse(fetchFn.mock.calls[1][1].body);
    expect(execBody.file_ids).toContain(555);
  });
});
