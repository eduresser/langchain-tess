/**
 * JSON-based tool calling support for Tess AI.
 *
 * Since the Tess API does not support native OpenAI-style tool calling,
 * this module provides prompt engineering utilities to enforce a JSON-only
 * response protocol and parse structured tool call responses.
 *
 * The model is always instructed to respond with a single JSON object:
 *   {"content": "...", "tool_calls": [...]}
 *
 * The closing `}` of the JSON object provides a natural stop boundary -
 * any hallucinated text after it is trivially discarded by extracting only
 * the first balanced JSON object from the raw output.
 */

import { randomUUID } from "node:crypto";
import type { ParsedToolCall } from "./types.js";

// ------------------------------------------------------------------
// Error
// ------------------------------------------------------------------

export class ToolCallParseError extends Error {
  rawOutput: string;

  constructor(message: string, rawOutput: string = "") {
    super(message);
    this.name = "ToolCallParseError";
    this.rawOutput = rawOutput;
  }
}

// ------------------------------------------------------------------
// System prompts (100% English, JSON-only protocol)
// ------------------------------------------------------------------

export const JSON_RESPONSE_SYSTEM_PROMPT = `You must ALWAYS respond with a single valid JSON object and nothing else.

Format:
{"content": "your response text"}

Rules:
- Your entire response MUST be exactly one JSON object.
- The "content" field is REQUIRED and must be a string containing your reply.
- Do NOT write any text, markdown, or explanation outside the JSON object.
- Do NOT wrap the JSON in code fences or backticks.`;

export const JSON_TOOL_CALLING_SYSTEM_PROMPT = `You have access to the following tools:

{tool_definitions}

You must ALWAYS respond with a single valid JSON object and nothing else.

When you need to call one or more tools:
{"content": "optional short explanation", "tool_calls": [{"name": "tool_name", "arguments": {"arg1": "value1"}}]}

When you do NOT need to call any tool:
{"content": "your response text"}

Rules:
- Your entire response MUST be exactly one JSON object.
- The "content" field is REQUIRED and must be a string.
- The "tool_calls" field is OPTIONAL. Include it ONLY when you need to invoke tools.
- Each tool call must have "name" (exactly matching a tool name above) and "arguments" (a JSON object matching the tool's parameter schema).
- You may call multiple tools at once by adding multiple entries to the "tool_calls" array.
- Do NOT write any text, markdown, or explanation outside the JSON object.
- Do NOT wrap the JSON in code fences or backticks.
- Do NOT simulate or imagine tool results. Stop after emitting the JSON object and wait.`;

// ------------------------------------------------------------------
// Formatting helpers
// ------------------------------------------------------------------

export function formatToolsForPrompt(
  tools: Array<Record<string, unknown>>,
): string {
  const parts: string[] = [];
  for (const tool of tools) {
    const func = (tool.function as Record<string, unknown>) ?? tool;
    const name = (func.name as string) ?? "unknown";
    const description =
      (func.description as string) ?? "No description provided.";
    const params = (func.parameters as Record<string, unknown>) ?? {};
    const properties =
      (params.properties as Record<string, Record<string, unknown>>) ?? {};
    const required = new Set(
      (params.required as string[] | undefined) ?? [],
    );

    const paramLines: string[] = [];
    for (const [pname, pschema] of Object.entries(properties)) {
      const ptype = (pschema.type as string) ?? "any";
      const pdesc = (pschema.description as string) ?? "";
      const reqMarker = required.has(pname) ? " (required)" : " (optional)";
      paramLines.push(`    - ${pname} (${ptype}${reqMarker}): ${pdesc}`);
    }

    const paramsBlock =
      paramLines.length > 0 ? paramLines.join("\n") : "    (no parameters)";
    parts.push(
      `Tool: ${name}\n  Description: ${description}\n  Parameters:\n${paramsBlock}`,
    );
  }
  return parts.join("\n\n");
}

export function buildToolChoiceInstruction(
  toolChoice: string | Record<string, unknown> | boolean | null | undefined,
  tools?: Array<Record<string, unknown>>,
): string {
  if (toolChoice == null || toolChoice === "auto") return "";

  if (toolChoice === "none" || toolChoice === false) {
    return (
      '\n\nIMPORTANT: Do NOT call any tools. ' +
      'Respond with content only. Do NOT include the "tool_calls" field.'
    );
  }

  if (
    toolChoice === "required" ||
    toolChoice === "any" ||
    toolChoice === true
  ) {
    return (
      "\n\nIMPORTANT: You MUST call at least one tool. " +
      'The "tool_calls" array is REQUIRED in your response.'
    );
  }

  if (typeof toolChoice === "object" && toolChoice !== null) {
    const funcObj = (toolChoice.function as Record<string, unknown>) ?? {};
    const name =
      (funcObj.name as string) ?? (toolChoice.name as string) ?? "";
    if (name) {
      return `\n\nIMPORTANT: You MUST call the tool "${name}". No other tool is allowed.`;
    }
  }

  if (typeof toolChoice === "string") {
    const knownNames = new Set<string>();
    for (const t of tools ?? []) {
      const func = (t.function as Record<string, unknown>) ?? t;
      knownNames.add((func.name as string) ?? "");
    }
    if (knownNames.has(toolChoice)) {
      return `\n\nIMPORTANT: You MUST call the tool "${toolChoice}". No other tool is allowed.`;
    }
  }

  return "";
}

export function buildJsonPrompt(
  tools?: Array<Record<string, unknown>> | null,
  toolChoice?: string | Record<string, unknown> | boolean | null,
): string {
  if (tools && tools.length > 0) {
    const prompt = JSON_TOOL_CALLING_SYSTEM_PROMPT.replace(
      "{tool_definitions}",
      formatToolsForPrompt(tools),
    );
    return prompt + buildToolChoiceInstruction(toolChoice, tools);
  }
  return JSON_RESPONSE_SYSTEM_PROMPT;
}

// ------------------------------------------------------------------
// JSON extraction
// ------------------------------------------------------------------

function findBalancedEnd(
  text: string,
  start: number,
  openCh: string,
  closeCh: string,
): number {
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      if (inString) escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === openCh) {
      depth++;
    } else if (ch === closeCh) {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

export function hasTrailingContent(text: string): boolean {
  const firstBrace = text.indexOf("{");
  const firstBracket = text.indexOf("[");

  const candidates: Array<[number, string, string]> = [];
  if (firstBrace !== -1) candidates.push([firstBrace, "{", "}"]);
  if (firstBracket !== -1) candidates.push([firstBracket, "[", "]"]);

  if (candidates.length === 0) return false;
  candidates.sort((a, b) => a[0] - b[0]);

  for (const [start, openCh, closeCh] of candidates) {
    const end = findBalancedEnd(text, start, openCh, closeCh);
    if (end === -1) continue;
    const remainder = text.slice(end + 1).trim();
    return remainder.length > 0;
  }
  return false;
}

export function parseJsonString(
  text: string,
): Record<string, unknown> | unknown[] | string {
  const firstBrace = text.indexOf("{");
  const firstBracket = text.indexOf("[");

  const candidates: Array<[number, string, string]> = [];
  if (firstBrace !== -1) candidates.push([firstBrace, "{", "}"]);
  if (firstBracket !== -1) candidates.push([firstBracket, "[", "]"]);

  if (candidates.length === 0) return text;
  candidates.sort((a, b) => a[0] - b[0]);

  for (const [start, openCh, closeCh] of candidates) {
    const end = findBalancedEnd(text, start, openCh, closeCh);
    if (end === -1) continue;
    const jsonStr = text.slice(start, end + 1);
    try {
      return JSON.parse(jsonStr) as Record<string, unknown> | unknown[];
    } catch {
      continue;
    }
  }
  return text;
}

export function deepParseJson(value: unknown): unknown {
  if (typeof value === "string") {
    const parsed = parseJsonString(value);
    if (
      parsed !== value &&
      (typeof parsed === "object" && parsed !== null)
    ) {
      return deepParseJson(parsed);
    }
    return parsed;
  }

  if (Array.isArray(value)) {
    return value.map((item) => deepParseJson(item));
  }

  if (typeof value === "object" && value !== null) {
    const result: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      result[k] = deepParseJson(v);
    }
    return result;
  }

  return value;
}

// ------------------------------------------------------------------
// Contract validation
// ------------------------------------------------------------------

export function validateToolCallContract(
  data: Record<string, unknown>,
  rawOutput: string = "",
): ParsedToolCall[] {
  const toolCalls = data.tool_calls;
  if (toolCalls == null) {
    throw new ToolCallParseError(
      'Missing "tool_calls" key in parsed JSON',
      rawOutput,
    );
  }
  if (!Array.isArray(toolCalls)) {
    throw new ToolCallParseError(
      `"tool_calls" must be a list, got ${typeof toolCalls}`,
      rawOutput,
    );
  }
  if (toolCalls.length === 0) {
    throw new ToolCallParseError('"tool_calls" array is empty', rawOutput);
  }

  const validated: ParsedToolCall[] = [];
  for (let i = 0; i < toolCalls.length; i++) {
    const tc = toolCalls[i] as Record<string, unknown>;
    if (typeof tc !== "object" || tc === null || Array.isArray(tc)) {
      throw new ToolCallParseError(
        `tool_calls[${i}] must be a dict, got ${typeof tc}`,
        rawOutput,
      );
    }
    const name = tc.name;
    if (typeof name !== "string" || !name) {
      throw new ToolCallParseError(
        `tool_calls[${i}] missing or invalid "name"`,
        rawOutput,
      );
    }
    const args = tc.arguments;
    if (typeof args !== "object" || args === null || Array.isArray(args)) {
      throw new ToolCallParseError(
        `tool_calls[${i}] "arguments" must be a dict, got ${typeof args}`,
        rawOutput,
      );
    }
    validated.push({
      name,
      args: args as Record<string, unknown>,
      id:
        (tc.id as string) ||
        `call_${randomUUID().replace(/-/g, "").slice(0, 12)}`,
    });
  }
  return validated;
}

// ------------------------------------------------------------------
// JSON response parsing
// ------------------------------------------------------------------

export function parseJsonResponse(
  text: string,
): [string, ParsedToolCall[] | null] {
  const result = parseJsonString(text);

  if (typeof result === "string") {
    throw new ToolCallParseError("Model response is not valid JSON", text);
  }

  if (Array.isArray(result)) {
    throw new ToolCallParseError(
      `Expected JSON object, got array`,
      text,
    );
  }

  const deepParsed = deepParseJson(result) as Record<string, unknown>;

  const content = deepParsed.content;
  if (content == null) {
    throw new ToolCallParseError(
      'Missing required "content" field in JSON response',
      text,
    );
  }
  if (typeof content !== "string") {
    throw new ToolCallParseError(
      `"content" must be a string, got ${typeof content}`,
      text,
    );
  }

  const rawToolCalls = deepParsed.tool_calls;
  if (rawToolCalls == null) {
    return [content, null];
  }

  if (Array.isArray(rawToolCalls) && rawToolCalls.length === 0) {
    return [content, null];
  }

  const validated = validateToolCallContract(deepParsed, text);
  return [content, validated];
}

// ------------------------------------------------------------------
// Incremental JSON content extractor (for SSE streaming)
// ------------------------------------------------------------------

enum ExtractorState {
  SEEKING_CONTENT,
  IN_CONTENT_STRING,
  AFTER_CONTENT,
  PASSTHROUGH,
}

const PASSTHROUGH_CHAR_THRESHOLD = 40;

export class IncrementalJsonContentExtractor {
  private _buffer: string[] = [];
  private _state: ExtractorState = ExtractorState.SEEKING_CONTENT;
  private _escapeNext = false;
  private _contentChars: string[] = [];
  private _scanPos = 0;
  private _passthroughThreshold: number;
  private _seenToolCalls = false;

  constructor(passthroughThreshold: number = PASSTHROUGH_CHAR_THRESHOLD) {
    this._passthroughThreshold = passthroughThreshold;
  }

  feed(chunk: string): string {
    this._buffer.push(chunk);

    if (this._state === ExtractorState.PASSTHROUGH) return chunk;
    if (this._state === ExtractorState.AFTER_CONTENT) return "";

    if (this._state === ExtractorState.SEEKING_CONTENT) {
      this._tryFindContentStart();
      if (this._state === ExtractorState.SEEKING_CONTENT) {
        if (this._shouldSwitchToPassthrough()) {
          this._state = ExtractorState.PASSTHROUGH;
          return this._fullBuffer();
        }
      }
    }

    if (this._state === ExtractorState.IN_CONTENT_STRING) {
      return this._extractContentChars();
    }

    return "";
  }

  get contentComplete(): boolean {
    return (
      this._state === ExtractorState.AFTER_CONTENT ||
      this._state === ExtractorState.PASSTHROUGH
    );
  }

  get isPassthrough(): boolean {
    return this._state === ExtractorState.PASSTHROUGH;
  }

  getFullText(): string {
    return this._buffer.join("");
  }

  getExtractedContent(): string {
    return this._contentChars.join("");
  }

  private _fullBuffer(): string {
    return this._buffer.join("");
  }

  private _shouldSwitchToPassthrough(): boolean {
    if (this._seenToolCalls) return false;
    return this._fullBuffer().length >= this._passthroughThreshold;
  }

  private _tryFindContentStart(): void {
    const buf = this._fullBuffer();
    const needle = '"content"';
    const tcNeedle = '"tool_calls"';
    let idx = buf.indexOf(needle, this._scanPos);
    const tcIdx = buf.indexOf(tcNeedle, this._scanPos);

    if (tcIdx !== -1 && (idx === -1 || tcIdx < idx)) {
      this._seenToolCalls = true;
      this._scanPos = tcIdx + tcNeedle.length;
      idx = buf.indexOf(needle, this._scanPos);
    }

    if (idx === -1) {
      this._scanPos = Math.max(0, buf.length - needle.length);
      return;
    }

    const colonIdx = buf.indexOf(":", idx + needle.length);
    if (colonIdx === -1) return;
    const quoteIdx = buf.indexOf('"', colonIdx + 1);
    if (quoteIdx === -1) return;

    this._state = ExtractorState.IN_CONTENT_STRING;
    this._scanPos = quoteIdx + 1;
  }

  private _extractContentChars(): string {
    const buf = this._fullBuffer();
    const newChars: string[] = [];
    let i = this._scanPos;
    while (i < buf.length) {
      const ch = buf[i];
      if (this._escapeNext) {
        if (ch === "n") newChars.push("\n");
        else if (ch === "t") newChars.push("\t");
        else if (ch === "r") newChars.push("\r");
        else if (ch === '"') newChars.push('"');
        else if (ch === "\\") newChars.push("\\");
        else if (ch === "/") newChars.push("/");
        else if (ch === "b") newChars.push("\b");
        else if (ch === "f") newChars.push("\f");
        else if (ch === "u" && i + 4 < buf.length) {
          const hexStr = buf.slice(i + 1, i + 5);
          try {
            newChars.push(String.fromCharCode(parseInt(hexStr, 16)));
          } catch {
            newChars.push("\\u" + hexStr);
          }
          i += 4;
        } else {
          newChars.push(ch);
        }
        this._escapeNext = false;
        i++;
        continue;
      }
      if (ch === "\\") {
        this._escapeNext = true;
        i++;
        continue;
      }
      if (ch === '"') {
        this._state = ExtractorState.AFTER_CONTENT;
        this._scanPos = i + 1;
        this._contentChars.push(...newChars);
        return newChars.join("");
      }
      newChars.push(ch);
      i++;
    }

    this._scanPos = i;
    this._contentChars.push(...newChars);
    return newChars.join("");
  }
}
