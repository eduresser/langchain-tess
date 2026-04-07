import { describe, it, expect } from "vitest";
import {
  ToolCallParseError,
  formatToolsForPrompt,
  buildToolChoiceInstruction,
  buildJsonPrompt,
  parseJsonString,
  deepParseJson,
  hasTrailingContent,
  validateToolCallContract,
  parseJsonResponse,
  IncrementalJsonContentExtractor,
  JSON_RESPONSE_SYSTEM_PROMPT,
} from "../tool_calling.js";

// ------------------------------------------------------------------
// parseJsonString
// ------------------------------------------------------------------

describe("parseJsonString", () => {
  it("parses a simple JSON object", () => {
    const result = parseJsonString('{"content": "hello"}');
    expect(result).toEqual({ content: "hello" });
  });

  it("extracts first balanced JSON from text with trailing content", () => {
    const result = parseJsonString('{"content": "hello"} some extra text');
    expect(result).toEqual({ content: "hello" });
  });

  it("extracts JSON with leading text", () => {
    const result = parseJsonString('some preamble {"content": "hello"}');
    expect(result).toEqual({ content: "hello" });
  });

  it("returns original string when no JSON found", () => {
    const result = parseJsonString("just plain text");
    expect(result).toBe("just plain text");
  });

  it("handles nested objects", () => {
    const result = parseJsonString(
      '{"content": "hi", "tool_calls": [{"name": "test", "arguments": {"key": "val"}}]}',
    );
    expect(result).toEqual({
      content: "hi",
      tool_calls: [{ name: "test", arguments: { key: "val" } }],
    });
  });

  it("handles JSON arrays", () => {
    const result = parseJsonString('[1, 2, 3] trailing');
    expect(result).toEqual([1, 2, 3]);
  });

  it("returns original on invalid JSON", () => {
    const result = parseJsonString("{broken json");
    expect(result).toBe("{broken json");
  });

  it("handles escaped quotes in strings", () => {
    const result = parseJsonString('{"content": "he said \\"hello\\""}');
    expect(result).toEqual({ content: 'he said "hello"' });
  });
});

// ------------------------------------------------------------------
// hasTrailingContent
// ------------------------------------------------------------------

describe("hasTrailingContent", () => {
  it("returns false for clean JSON", () => {
    expect(hasTrailingContent('{"content": "hello"}')).toBe(false);
  });

  it("returns true when text follows JSON", () => {
    expect(hasTrailingContent('{"content": "hello"} extra')).toBe(true);
  });

  it("returns false for JSON with only whitespace after", () => {
    expect(hasTrailingContent('{"content": "hello"}   \n  ')).toBe(false);
  });

  it("returns false for no JSON at all", () => {
    expect(hasTrailingContent("plain text")).toBe(false);
  });

  it("returns true for hallucinated continuation", () => {
    expect(
      hasTrailingContent(
        '{"content": "hello"}\n{"content": "second object"}',
      ),
    ).toBe(true);
  });
});

// ------------------------------------------------------------------
// deepParseJson
// ------------------------------------------------------------------

describe("deepParseJson", () => {
  it("parses stringified JSON in values", () => {
    const result = deepParseJson({ args: '{"city": "SP"}' });
    expect(result).toEqual({ args: { city: "SP" } });
  });

  it("recurses into arrays", () => {
    const result = deepParseJson(['{"key": "val"}']);
    expect(result).toEqual([{ key: "val" }]);
  });

  it("leaves non-JSON strings alone", () => {
    expect(deepParseJson("hello")).toBe("hello");
  });

  it("leaves numbers alone", () => {
    expect(deepParseJson(42)).toBe(42);
  });

  it("leaves null alone", () => {
    expect(deepParseJson(null)).toBe(null);
  });
});

// ------------------------------------------------------------------
// validateToolCallContract
// ------------------------------------------------------------------

describe("validateToolCallContract", () => {
  it("validates correct tool calls", () => {
    const data = {
      content: "calling tool",
      tool_calls: [{ name: "test", arguments: { city: "SP" } }],
    };
    const result = validateToolCallContract(data);
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe("test");
    expect(result[0].args).toEqual({ city: "SP" });
    expect(result[0].id).toMatch(/^call_/);
  });

  it("preserves existing id", () => {
    const data = {
      tool_calls: [
        { name: "test", arguments: { city: "SP" }, id: "my_custom_id" },
      ],
    };
    const result = validateToolCallContract(data);
    expect(result[0].id).toBe("my_custom_id");
  });

  it("throws when tool_calls is missing", () => {
    expect(() => validateToolCallContract({ content: "hi" })).toThrow(
      ToolCallParseError,
    );
  });

  it("throws when tool_calls is not an array", () => {
    expect(() =>
      validateToolCallContract({ tool_calls: "not an array" }),
    ).toThrow(ToolCallParseError);
  });

  it("throws when tool_calls is empty", () => {
    expect(() =>
      validateToolCallContract({ tool_calls: [] }),
    ).toThrow(ToolCallParseError);
  });

  it("throws when name is missing", () => {
    expect(() =>
      validateToolCallContract({
        tool_calls: [{ arguments: {} }],
      }),
    ).toThrow(ToolCallParseError);
  });

  it("throws when arguments is not an object", () => {
    expect(() =>
      validateToolCallContract({
        tool_calls: [{ name: "test", arguments: "not an object" }],
      }),
    ).toThrow(ToolCallParseError);
  });
});

// ------------------------------------------------------------------
// parseJsonResponse
// ------------------------------------------------------------------

describe("parseJsonResponse", () => {
  it("parses content-only response", () => {
    const [content, toolCalls] = parseJsonResponse(
      '{"content": "Hello!"}',
    );
    expect(content).toBe("Hello!");
    expect(toolCalls).toBeNull();
  });

  it("parses response with tool calls", () => {
    const [content, toolCalls] = parseJsonResponse(
      '{"content": "calling", "tool_calls": [{"name": "get_weather", "arguments": {"city": "SP"}}]}',
    );
    expect(content).toBe("calling");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls![0].name).toBe("get_weather");
    expect(toolCalls![0].args).toEqual({ city: "SP" });
  });

  it("returns null tool_calls for empty array", () => {
    const [content, toolCalls] = parseJsonResponse(
      '{"content": "no tools", "tool_calls": []}',
    );
    expect(content).toBe("no tools");
    expect(toolCalls).toBeNull();
  });

  it("ignores trailing content after JSON", () => {
    const [content] = parseJsonResponse(
      '{"content": "hello"} some hallucinated text',
    );
    expect(content).toBe("hello");
  });

  it("throws for non-JSON text", () => {
    expect(() => parseJsonResponse("not json")).toThrow(ToolCallParseError);
  });

  it("throws when content field is missing", () => {
    expect(() => parseJsonResponse('{"foo": "bar"}')).toThrow(
      ToolCallParseError,
    );
  });

  it("throws when content is not a string", () => {
    expect(() => parseJsonResponse('{"content": 42}')).toThrow(
      ToolCallParseError,
    );
  });

  it("carries rawOutput on parse error", () => {
    try {
      parseJsonResponse("invalid json");
      expect.fail("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ToolCallParseError);
      expect((err as ToolCallParseError).rawOutput).toBe("invalid json");
    }
  });
});

// ------------------------------------------------------------------
// formatToolsForPrompt
// ------------------------------------------------------------------

describe("formatToolsForPrompt", () => {
  it("formats a simple tool", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the weather",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string", description: "City name" },
            },
            required: ["city"],
          },
        },
      },
    ];
    const result = formatToolsForPrompt(tools);
    expect(result).toContain("Tool: get_weather");
    expect(result).toContain("Description: Get the weather");
    expect(result).toContain("city (string (required)): City name");
  });

  it("handles tools without function wrapper", () => {
    const tools = [
      {
        name: "simple_tool",
        description: "A simple tool",
        parameters: {},
      },
    ];
    const result = formatToolsForPrompt(tools);
    expect(result).toContain("Tool: simple_tool");
  });

  it("marks optional parameters", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "test",
          description: "test",
          parameters: {
            properties: {
              opt: { type: "string" },
            },
            required: [],
          },
        },
      },
    ];
    const result = formatToolsForPrompt(tools);
    expect(result).toContain("(optional)");
  });
});

// ------------------------------------------------------------------
// buildToolChoiceInstruction
// ------------------------------------------------------------------

describe("buildToolChoiceInstruction", () => {
  it('returns empty for null / "auto"', () => {
    expect(buildToolChoiceInstruction(null)).toBe("");
    expect(buildToolChoiceInstruction("auto")).toBe("");
  });

  it('returns "do not call" for "none"', () => {
    const result = buildToolChoiceInstruction("none");
    expect(result).toContain("Do NOT call any tools");
  });

  it('returns "do not call" for false', () => {
    const result = buildToolChoiceInstruction(false);
    expect(result).toContain("Do NOT call any tools");
  });

  it('returns "must call" for "required"', () => {
    const result = buildToolChoiceInstruction("required");
    expect(result).toContain("MUST call at least one tool");
  });

  it('returns "must call" for true', () => {
    const result = buildToolChoiceInstruction(true);
    expect(result).toContain("MUST call at least one tool");
  });

  it("returns specific tool for dict with function.name", () => {
    const result = buildToolChoiceInstruction({
      function: { name: "my_tool" },
    });
    expect(result).toContain('MUST call the tool "my_tool"');
  });

  it("returns specific tool for string matching known tool name", () => {
    const tools = [
      { type: "function", function: { name: "get_weather" } },
    ];
    const result = buildToolChoiceInstruction("get_weather", tools);
    expect(result).toContain('MUST call the tool "get_weather"');
  });

  it("returns empty for unknown string", () => {
    const result = buildToolChoiceInstruction("unknown_thing", []);
    expect(result).toBe("");
  });
});

// ------------------------------------------------------------------
// buildJsonPrompt
// ------------------------------------------------------------------

describe("buildJsonPrompt", () => {
  it("returns simple prompt when no tools", () => {
    const result = buildJsonPrompt();
    expect(result).toBe(JSON_RESPONSE_SYSTEM_PROMPT);
  });

  it("returns tool prompt when tools provided", () => {
    const tools = [
      {
        type: "function",
        function: {
          name: "test",
          description: "test tool",
          parameters: {},
        },
      },
    ];
    const result = buildJsonPrompt(tools);
    expect(result).toContain("Tool: test");
    expect(result).toContain("tool_calls");
  });

  it("includes tool_choice instruction when provided", () => {
    const tools = [
      { type: "function", function: { name: "test", description: "d", parameters: {} } },
    ];
    const result = buildJsonPrompt(tools, "required");
    expect(result).toContain("MUST call at least one tool");
  });
});

// ------------------------------------------------------------------
// IncrementalJsonContentExtractor
// ------------------------------------------------------------------

describe("IncrementalJsonContentExtractor", () => {
  it("extracts content from streamed JSON", () => {
    const ext = new IncrementalJsonContentExtractor();
    let result = "";
    result += ext.feed('{"con');
    result += ext.feed('tent": "hel');
    result += ext.feed('lo world"}');
    expect(result).toBe("hello world");
    expect(ext.contentComplete).toBe(true);
    expect(ext.isPassthrough).toBe(false);
  });

  it("handles escape sequences", () => {
    const ext = new IncrementalJsonContentExtractor();
    let result = "";
    result += ext.feed('{"content": "line1\\nline2"}');
    expect(result).toBe("line1\nline2");
  });

  it("handles unicode escapes", () => {
    const ext = new IncrementalJsonContentExtractor();
    let result = "";
    result += ext.feed('{"content": "caf\\u00e9"}');
    expect(result).toBe("café");
  });

  it("switches to passthrough for non-JSON text", () => {
    const ext = new IncrementalJsonContentExtractor(10);
    const result = ext.feed("This is plain text with more chars");
    expect(result).toBe("This is plain text with more chars");
    expect(ext.isPassthrough).toBe(true);
    expect(ext.contentComplete).toBe(true);
  });

  it("stays in seeking mode until threshold", () => {
    const ext = new IncrementalJsonContentExtractor(100);
    const result = ext.feed("short");
    expect(result).toBe("");
    expect(ext.isPassthrough).toBe(false);
  });

  it("handles tool_calls before content", () => {
    const ext = new IncrementalJsonContentExtractor();
    let result = "";
    result += ext.feed(
      '{"tool_calls": [{"name": "test", "arguments": {}}], "content": "done"}',
    );
    expect(result).toBe("done");
    expect(ext.contentComplete).toBe(true);
  });

  it("does not passthrough when tool_calls seen", () => {
    const ext = new IncrementalJsonContentExtractor(10);
    let result = "";
    result += ext.feed('{"tool_calls":');
    // Even though we're past threshold, tool_calls was seen
    result += ext.feed(' [], "content": "hi"}');
    expect(result).toBe("hi");
    expect(ext.isPassthrough).toBe(false);
  });

  it("returns empty after content is complete", () => {
    const ext = new IncrementalJsonContentExtractor();
    ext.feed('{"content": "done"}');
    expect(ext.feed("more data")).toBe("");
  });

  it("getFullText returns complete buffer", () => {
    const ext = new IncrementalJsonContentExtractor();
    ext.feed("chunk1");
    ext.feed("chunk2");
    expect(ext.getFullText()).toBe("chunk1chunk2");
  });

  it("getExtractedContent returns content chars", () => {
    const ext = new IncrementalJsonContentExtractor();
    ext.feed('{"content": "hello"}');
    expect(ext.getExtractedContent()).toBe("hello");
  });

  it("passthrough returns all accumulated text at once", () => {
    const ext = new IncrementalJsonContentExtractor(5);
    const r1 = ext.feed("Hello world!");
    expect(ext.isPassthrough).toBe(true);
    expect(r1).toBe("Hello world!");
    const r2 = ext.feed(" More text");
    expect(r2).toBe(" More text");
  });
});
