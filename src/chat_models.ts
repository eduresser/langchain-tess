/**
 * Tess AI chat model integration for LangChain.js
 */

import { createHash } from "node:crypto";
import { getEncoding } from "js-tiktoken";
import {
  BaseChatModel,
  type BaseChatModelParams,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { type Runnable, RunnableLambda } from "@langchain/core/runnables";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";

import {
  TessAPIError,
  TessAuthenticationError,
  TessPayloadTooLargeError,
  TessRateLimitError,
  raiseForTessStatus,
} from "./exceptions.js";
import {
  IncrementalJsonContentExtractor,
  ToolCallParseError,
  buildJsonPrompt,
  hasTrailingContent,
  parseJsonResponse,
} from "./tool_calling.js";
import type {
  ChatTessAIInput,
  ChatTessAICallOptions,
  FileRef,
  TessMessage,
  TessResponseMetadata,
} from "./types.js";

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------

const MIME_TO_EXTENSION: Record<string, string> = {
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
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    ".docx",
  "application/msword": ".doc",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation":
    ".pptx",
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
};

const FILE_TERMINAL_STATUSES = new Set(["completed", "failed"]);
const TERMINAL_STATUSES = new Set([
  "succeeded",
  "failed",
  "error",
  "completed",
]);

// ------------------------------------------------------------------
// FileRef helpers
// ------------------------------------------------------------------

function getExtension(mimeType: string): string {
  const ext = MIME_TO_EXTENSION[mimeType];
  if (ext) return ext;
  return ".bin";
}

function computeContentHash(data: Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

function getUploadFilename(ref: FileRef): string {
  const h = ref.data ? computeContentHash(ref.data).slice(0, 12) : "file";
  return `${h}${getExtension(ref.mimeType)}`;
}

// ------------------------------------------------------------------
// Sleep helper
// ------------------------------------------------------------------

function sleep(seconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, seconds * 1000));
}

// ------------------------------------------------------------------
// Main class
// ------------------------------------------------------------------

export class ChatTessAI extends BaseChatModel<ChatTessAICallOptions> {
  static lc_name() {
    return "ChatTessAI";
  }

  lc_serializable = true;

  get lc_secrets(): Record<string, string> {
    return { apiKey: "TESS_API_KEY" };
  }

  get lc_aliases(): Record<string, string> {
    return { apiKey: "tess_api_key" };
  }

  apiKey: string;
  agentId: number;
  model: string;
  temperature: number;
  tools: string;
  workspaceId: number;
  baseUrl: string;
  timeout: number;
  maxRetries: number;
  waitExecution: boolean;
  pollingInterval: number;
  fileIds?: number[];
  trackConversations: boolean;
  maxTrackedConversations: number;

  private _conversationCache: Map<string, [number, number]> = new Map();
  private _cacheOrder: string[] = [];
  private _fileCache: Map<string, number> = new Map();

  constructor(fields: ChatTessAIInput & BaseChatModelParams) {
    super(fields);
    this.apiKey =
      fields.apiKey ?? (typeof process !== "undefined" ? process.env?.TESS_API_KEY ?? "" : "");
    this.agentId = fields.agentId;
    this.model = fields.model ?? "tess-5";
    this.temperature = fields.temperature ?? 1.0;
    this.tools = fields.tools ?? "no-tools";
    this.workspaceId = fields.workspaceId;
    this.baseUrl = fields.baseUrl ?? "https://api.tess.im";
    this.timeout = fields.timeout ?? 120;
    this.maxRetries = fields.maxRetries ?? 5;
    this.waitExecution = fields.waitExecution ?? true;
    this.pollingInterval = fields.pollingInterval ?? 5.0;
    this.fileIds = fields.fileIds;
    this.trackConversations = fields.trackConversations ?? true;
    this.maxTrackedConversations = fields.maxTrackedConversations ?? 100;
  }

  _llmType(): string {
    return "tess-ai";
  }

  _identifyingParams(): Record<string, unknown> {
    return {
      model_name: this.model,
      agent_id: this.agentId,
      temperature: this.temperature,
      tools: this.tools,
    };
  }

  // ------------------------------------------------------------------
  // Conversation tracking helpers
  // ------------------------------------------------------------------

  private static _hashMessages(messages: TessMessage[]): string {
    const serialized = JSON.stringify(messages);
    return createHash("sha256").update(serialized).digest("hex");
  }

  private _findConversation(
    converted: TessMessage[],
  ): [number | null, number] {
    for (let k = converted.length - 1; k > 0; k--) {
      const h = ChatTessAI._hashMessages(converted.slice(0, k));
      const cached = this._conversationCache.get(h);
      if (cached) {
        return [cached[0], k];
      }
    }
    return [null, 0];
  }

  private _updateConversationCache(
    fullMessages: TessMessage[],
    rootId: number,
  ): void {
    const h = ChatTessAI._hashMessages(fullMessages);
    const existingIdx = this._cacheOrder.indexOf(h);
    if (existingIdx !== -1) {
      this._cacheOrder.splice(existingIdx, 1);
    }
    this._conversationCache.set(h, [rootId, fullMessages.length]);
    this._cacheOrder.push(h);
    while (this._cacheOrder.length > this.maxTrackedConversations) {
      const evicted = this._cacheOrder.shift()!;
      this._conversationCache.delete(evicted);
    }
  }

  private _trackAfterResponse(
    converted: TessMessage[],
    assistantMsg: AIMessage,
    metadata: TessResponseMetadata,
    rawOutput?: string,
  ): void {
    if (!this.trackConversations) return;
    const newRootId = metadata.tess_root_id;
    if (newRootId == null) return;
    if (rawOutput != null && hasTrailingContent(rawOutput)) return;
    const responseMsg: TessMessage = {
      role: "assistant",
      content: ChatTessAI._assistantMessageToTessContent(assistantMsg),
    };
    const fullConversation = [...converted, responseMsg];
    this._updateConversationCache(fullConversation, newRootId);
  }

  private _invalidateRootId(rootId: number): void {
    const toRemove: string[] = [];
    for (const [h, [rid]] of this._conversationCache) {
      if (rid === rootId) toRemove.push(h);
    }
    for (const h of toRemove) {
      this._conversationCache.delete(h);
      const idx = this._cacheOrder.indexOf(h);
      if (idx !== -1) this._cacheOrder.splice(idx, 1);
    }
  }

  resetConversations(): void {
    this._conversationCache.clear();
    this._cacheOrder = [];
  }

  // ------------------------------------------------------------------
  // Token counting (js-tiktoken)
  // ------------------------------------------------------------------

  private _getEncoding() {
    if (this.model.startsWith("gpt-")) {
      return getEncoding("cl100k_base");
    }
    return getEncoding("p50k_base");
  }

  getNumTokens(text: string): Promise<number> {
    const enc = this._getEncoding();
    return Promise.resolve(enc.encode(text).length);
  }

  // ------------------------------------------------------------------
  // bind_tools / with_structured_output
  // ------------------------------------------------------------------

  bindTools(
    tools: BindToolsInput[],
    kwargs?: Partial<ChatTessAICallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatTessAICallOptions> {
    const formatted = tools.map((t) =>
      convertToOpenAITool(t as Record<string, unknown>),
    );
    return this.bind({
      tools: formatted,
      ...kwargs,
    } as Partial<ChatTessAICallOptions>);
  }

  withStructuredOutput<T = unknown>(
    schema: Record<string, unknown> | { new (...args: unknown[]): T },
    config?: { includeRaw?: boolean },
  ): Runnable {
    const includeRaw = config?.includeRaw ?? false;

    if (typeof schema === "function") {
      const SchemaClass = schema;
      const toolDef = convertToOpenAITool(schema as unknown as Record<string, unknown>);
      const toolName =
        (toolDef as unknown as { function: { name: string } }).function?.name;
      const llmWithTool = this.bindTools([schema as unknown as BindToolsInput]);

      const parse = new RunnableLambda({
        func: (msg: BaseMessage) => {
          if (!(msg instanceof AIMessage) || !msg.tool_calls?.length) {
            if (includeRaw)
              return { raw: msg, parsed: null, parsing_error: null };
            return null;
          }
          for (const tc of msg.tool_calls) {
            if (tc.name === toolName) {
              const parsed = new (SchemaClass as new (
                ...a: unknown[]
              ) => T)(tc.args);
              if (includeRaw)
                return { raw: msg, parsed, parsing_error: null };
              return parsed;
            }
          }
          if (includeRaw)
            return { raw: msg, parsed: null, parsing_error: null };
          return null;
        },
      });

      return llmWithTool.pipe(parse);
    }

    if (typeof schema === "object") {
      const toolName =
        (schema.title as string) ??
        (schema.name as string) ??
        "structured_output";
      const toolDef = {
        type: "function",
        function: {
          name: toolName,
          description: (schema.description as string) ?? "",
          parameters: schema,
        },
      };
      const llmWithTool = this.bind({
        tools: [toolDef],
      } as Partial<ChatTessAICallOptions>);

      const parseDict = new RunnableLambda({
        func: (msg: BaseMessage) => {
          if (!(msg instanceof AIMessage) || !msg.tool_calls?.length) {
            if (includeRaw)
              return { raw: msg, parsed: null, parsing_error: null };
            return null;
          }
          for (const tc of msg.tool_calls) {
            if (tc.name === toolName) {
              if (includeRaw)
                return { raw: msg, parsed: tc.args, parsing_error: null };
              return tc.args;
            }
          }
          if (includeRaw)
            return { raw: msg, parsed: null, parsing_error: null };
          return null;
        },
      });

      return llmWithTool.pipe(parseDict);
    }

    throw new TypeError(
      `schema must be a class or a dict, got ${typeof schema}`,
    );
  }

  // ------------------------------------------------------------------
  // Content helpers
  // ------------------------------------------------------------------

  private static _contentToStr(raw: unknown): string {
    if (typeof raw === "string") return raw;
    if (Array.isArray(raw)) {
      const parts: string[] = [];
      for (const block of raw) {
        if (typeof block === "object" && block !== null) {
          parts.push(
            (block as Record<string, string>).text ?? "",
          );
        } else if (typeof block === "string") {
          parts.push(block);
        }
      }
      return parts.join("\n");
    }
    return String(raw);
  }

  private static _assistantMessageToTessContent(msg: AIMessage): string {
    const result: Record<string, unknown> = {
      content: ChatTessAI._contentToStr(msg.content),
    };
    if (msg.tool_calls?.length) {
      result.tool_calls = msg.tool_calls.map((tc) => ({
        name: tc.name,
        arguments: tc.args,
      }));
    }
    return JSON.stringify(result);
  }

  private static _applyStopSequences(
    content: string,
    stop?: string[],
  ): string {
    if (!stop?.length) return content;
    let earliest = content.length;
    for (const seq of stop) {
      const idx = content.indexOf(seq);
      if (idx !== -1 && idx < earliest) earliest = idx;
    }
    return content.slice(0, earliest);
  }

  private _outputToAssistantMessage(
    output: string,
    _boundTools: Array<Record<string, unknown>> | undefined,
    metadata: TessResponseMetadata,
    stop?: string[],
  ): AIMessage {
    const [content, toolCalls] = parseJsonResponse(output);
    const truncated = ChatTessAI._applyStopSequences(content, stop);
    if (toolCalls) {
      return new AIMessage({
        content: truncated,
        tool_calls: toolCalls,
        response_metadata: metadata,
      });
    }
    return new AIMessage({ content: truncated, response_metadata: metadata });
  }

  // ------------------------------------------------------------------
  // JSON format prompt injection
  // ------------------------------------------------------------------

  private static _injectJsonFormatPrompt(
    messages: BaseMessage[],
    boundTools?: Array<Record<string, unknown>> | null,
    toolChoice?: unknown,
  ): BaseMessage[] {
    const jsonPrompt = buildJsonPrompt(
      boundTools,
      toolChoice as string | Record<string, unknown> | boolean | null | undefined,
    );

    if (messages.length > 0 && messages[0] instanceof SystemMessage) {
      const combined = `${jsonPrompt}\n\n${messages[0].content}`;
      return [new SystemMessage(combined), ...messages.slice(1)];
    }

    return [new SystemMessage(jsonPrompt), ...messages];
  }

  // ------------------------------------------------------------------
  // Message conversion
  // ------------------------------------------------------------------

  private static _convertMessages(messages: BaseMessage[]): TessMessage[] {
    const converted: TessMessage[] = [];
    for (const msg of messages) {
      let role: string;
      let content: string;

      if (msg instanceof SystemMessage) {
        role = "developer";
        content = ChatTessAI._contentToStr(msg.content);
      } else if (msg instanceof HumanMessage) {
        role = "user";
        content = ChatTessAI._contentToStr(msg.content);
      } else if (msg instanceof AIMessage) {
        role = "assistant";
        content = ChatTessAI._assistantMessageToTessContent(msg);
      } else if (msg instanceof ToolMessage) {
        const toolName = msg.name ?? "unknown_tool";
        role = "user";
        content = `[Tool Result] The tool "${toolName}" returned:\n${ChatTessAI._contentToStr(msg.content)}`;
      } else {
        role = "user";
        content = ChatTessAI._contentToStr(msg.content);
      }

      if (
        converted.length > 0 &&
        converted[converted.length - 1].role === role &&
        role !== "developer"
      ) {
        converted[converted.length - 1].content += `\n\n${content}`;
      } else {
        converted.push({ role, content });
      }
    }
    return converted;
  }

  // ------------------------------------------------------------------
  // Multimodal file extraction
  // ------------------------------------------------------------------

  private static _extractFileRefsFromContent(
    content: unknown[],
    msgIndex: number,
  ): [string[], FileRef[]] {
    const textParts: string[] = [];
    const fileRefs: FileRef[] = [];

    for (const block of content) {
      if (typeof block === "string") {
        textParts.push(block);
        continue;
      }
      if (typeof block !== "object" || block === null) continue;
      const b = block as Record<string, unknown>;
      const blockType = b.type as string;

      if (blockType === "text") {
        textParts.push((b.text as string) ?? "");
      } else if (blockType === "image" || blockType === "file") {
        const rawData = b.data as string;
        const mime = (b.mimeType as string) ?? "application/octet-stream";
        if (rawData) {
          const buffer = Buffer.from(rawData, "base64");
          fileRefs.push({
            data: new Uint8Array(buffer),
            mimeType: mime,
            messageIndex: msgIndex,
          });
        }
      } else if (blockType === "tess_ai") {
        fileRefs.push({
          fileId: b.file_id as number,
          mimeType: "",
          messageIndex: msgIndex,
        });
      } else if (blockType === "url") {
        fileRefs.push({
          url: b.url as string,
          mimeType: (b.mimeType as string) ?? "",
          messageIndex: msgIndex,
        });
      }
    }
    return [textParts, fileRefs];
  }

  private static _convertMessagesWithFiles(
    messages: BaseMessage[],
  ): [TessMessage[], FileRef[]] {
    const converted: TessMessage[] = [];
    const allFileRefs: FileRef[] = [];

    for (let idx = 0; idx < messages.length; idx++) {
      const msg = messages[idx];
      let role: string;
      let content: string;

      if (
        msg instanceof HumanMessage &&
        Array.isArray(msg.content)
      ) {
        const [textParts, fileRefs] = ChatTessAI._extractFileRefsFromContent(
          msg.content,
          idx,
        );
        allFileRefs.push(...fileRefs);
        role = "user";
        content = textParts.join("\n").trim() || "[See attached files]";
      } else if (msg instanceof SystemMessage) {
        role = "developer";
        content = ChatTessAI._contentToStr(msg.content);
      } else if (msg instanceof HumanMessage) {
        role = "user";
        content = ChatTessAI._contentToStr(msg.content);
      } else if (msg instanceof AIMessage) {
        role = "assistant";
        content = ChatTessAI._assistantMessageToTessContent(msg);
      } else if (msg instanceof ToolMessage) {
        const toolName = msg.name ?? "unknown_tool";
        role = "user";
        content = `[Tool Result] The tool "${toolName}" returned:\n${ChatTessAI._contentToStr(msg.content)}`;
      } else {
        role = "user";
        content = ChatTessAI._contentToStr(msg.content);
      }

      if (
        converted.length > 0 &&
        converted[converted.length - 1].role === role &&
        role !== "developer"
      ) {
        converted[converted.length - 1].content += `\n\n${content}`;
      } else {
        converted.push({ role, content });
      }
    }
    return [converted, allFileRefs];
  }

  // ------------------------------------------------------------------
  // Payload / request helpers
  // ------------------------------------------------------------------

  private get _executeUrl(): string {
    return `${this.baseUrl}/agents/${this.agentId}/execute`;
  }

  private _responseUrl(responseId: number): string {
    return `${this.baseUrl}/agent-responses/${responseId}`;
  }

  private get _headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
      "x-workspace-id": String(this.workspaceId),
    };
  }

  private get _uploadHeaders(): Record<string, string> {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      "x-workspace-id": String(this.workspaceId),
    };
  }

  private _buildPayloadFromConverted(
    convertedMessages: TessMessage[],
    options: {
      stream?: boolean;
      rootId?: number | null;
      fileIds?: number[] | null;
    } = {},
  ): Record<string, unknown> {
    const { stream = false, rootId, fileIds } = options;
    const payload: Record<string, unknown> = {
      model: this.model,
      temperature: `${this.temperature}`,
      tools: this.tools,
      messages: convertedMessages,
      wait_execution: stream ? false : this.waitExecution,
    };
    if (rootId != null) payload.root_id = rootId;
    if (stream) payload.stream = true;
    const effectiveFileIds = fileIds ?? this.fileIds;
    if (effectiveFileIds?.length) payload.file_ids = effectiveFileIds;
    return payload;
  }

  // ------------------------------------------------------------------
  // File upload + processing
  // ------------------------------------------------------------------

  private async _uploadAndProcessFile(
    fileBytes: Uint8Array,
    filename: string,
  ): Promise<number> {
    const formData = new FormData();
    formData.append(
      "file",
      new Blob([fileBytes.buffer as ArrayBuffer]),
      filename,
    );
    formData.append("process", "true");

    const resp = await fetch(`${this.baseUrl}/files`, {
      method: "POST",
      headers: this._uploadHeaders,
      body: formData,
      signal: AbortSignal.timeout(this.timeout * 1000),
    });

    if (!resp.ok) {
      throw new TessAPIError(
        `File upload failed: ${resp.status}`,
        resp.status,
      );
    }
    const fileData = (await resp.json()) as Record<string, unknown>;
    const fileId = fileData.id as number;
    const status = (fileData.status as string) ?? "";

    if (FILE_TERMINAL_STATUSES.has(status)) {
      if (status === "failed") {
        throw new Error(
          `Tess file processing failed for file_id=${fileId}: ${JSON.stringify(fileData)}`,
        );
      }
      return fileId;
    }

    const pollUrl = `${this.baseUrl}/files/${fileId}`;
    while (true) {
      await sleep(this.pollingInterval);
      const pollResp = await fetch(pollUrl, {
        headers: this._uploadHeaders,
        signal: AbortSignal.timeout(this.timeout * 1000),
      });
      if (!pollResp.ok) {
        throw new TessAPIError(
          `File poll failed: ${pollResp.status}`,
          pollResp.status,
        );
      }
      const pollData = (await pollResp.json()) as Record<string, unknown>;
      const pollStatus = (pollData.status as string) ?? "";

      if (pollStatus === "completed") return fileId;
      if (pollStatus === "failed") {
        throw new Error(
          `Tess file processing failed for file_id=${fileId}: ${JSON.stringify(pollData)}`,
        );
      }
    }
  }

  // ------------------------------------------------------------------
  // File ID resolution
  // ------------------------------------------------------------------

  private async _resolveFileIds(fileRefs: FileRef[]): Promise<number[]> {
    const fileIds: number[] = [];

    for (const ref of fileRefs) {
      if (ref.fileId != null) {
        fileIds.push(ref.fileId);
        continue;
      }

      if (ref.url && !ref.data) {
        const dlResp = await fetch(ref.url, {
          signal: AbortSignal.timeout(this.timeout * 1000),
        });
        if (!dlResp.ok) {
          throw new Error(`Failed to download file from ${ref.url}`);
        }
        ref.data = new Uint8Array(await dlResp.arrayBuffer());
        if (!ref.mimeType) {
          const ct =
            dlResp.headers.get("content-type") ?? "application/octet-stream";
          ref.mimeType = ct.split(";")[0].trim();
        }
      }

      if (ref.data) {
        const hash = computeContentHash(ref.data);
        const cached = this._fileCache.get(hash);
        if (cached != null) {
          fileIds.push(cached);
          continue;
        }

        const fid = await this._uploadAndProcessFile(
          ref.data,
          getUploadFilename(ref),
        );
        this._fileCache.set(hash, fid);
        fileIds.push(fid);
      }
    }

    return fileIds;
  }

  private static _mergeFileIds(
    staticIds: number[] | undefined,
    dynamicIds: number[],
  ): number[] | undefined {
    const combined = [...new Set([...(staticIds ?? []), ...dynamicIds])];
    return combined.length > 0 ? combined : undefined;
  }

  // ------------------------------------------------------------------
  // HTTP execution helpers
  // ------------------------------------------------------------------

  private async _fetchJson(
    url: string,
    init: RequestInit,
  ): Promise<[Record<string, unknown>, number]> {
    const resp = await fetch(url, {
      ...init,
      signal: init.signal ?? AbortSignal.timeout(this.timeout * 1000),
    });

    let body: Record<string, unknown> | undefined;
    try {
      body = (await resp.json()) as Record<string, unknown>;
    } catch {
      body = undefined;
    }

    if (resp.status >= 400) {
      raiseForTessStatus(resp.status, body);
    }

    return [body ?? {}, resp.status];
  }

  private async _executeRequest(
    payload: Record<string, unknown>,
  ): Promise<[string, TessResponseMetadata]> {
    const [data] = await this._fetchJson(this._executeUrl, {
      method: "POST",
      headers: this._headers,
      body: JSON.stringify(payload),
    });

    if (this.waitExecution && ChatTessAI._isCompleted(data)) {
      return ChatTessAI._extractOutputAndMetadata(data, this.model);
    }

    const responseId = ChatTessAI._extractResponseId(data);
    const pollResult = await this._pollForResult(responseId);
    return ChatTessAI._extractOutputAndMetadataFromPoll(
      pollResult,
      this.model,
    );
  }

  // ------------------------------------------------------------------
  // Polling
  // ------------------------------------------------------------------

  private static _isCompleted(data: Record<string, unknown>): boolean {
    const responses = data.responses as Array<Record<string, unknown>>;
    if (responses?.length) {
      const status = responses[0].status as string;
      return status === "succeeded" || status === "completed";
    }
    return false;
  }

  private static _extractResponseId(data: Record<string, unknown>): number {
    const responses = data.responses as Array<Record<string, unknown>>;
    if (responses?.length) return responses[0].id as number;
    throw new Error(`No response id found in API response: ${JSON.stringify(data)}`);
  }

  private async _pollForResult(
    responseId: number,
  ): Promise<Record<string, unknown>> {
    const url = this._responseUrl(responseId);
    while (true) {
      const [data] = await this._fetchJson(url, {
        method: "GET",
        headers: this._headers,
      });
      const status = data.status as string;
      if (TERMINAL_STATUSES.has(status)) {
        if (status === "failed" || status === "error") {
          throw new Error(
            `Tess AI execution failed: ${(data.error as string) ?? "unknown error"}`,
          );
        }
        return data;
      }
      await sleep(this.pollingInterval);
    }
  }

  // ------------------------------------------------------------------
  // Response extraction
  // ------------------------------------------------------------------

  private static _extractOutputAndMetadata(
    data: Record<string, unknown>,
    model: string,
  ): [string, TessResponseMetadata] {
    const responses = data.responses as Array<Record<string, unknown>>;
    if (!responses?.length) {
      throw new Error(`Empty responses from Tess API: ${JSON.stringify(data)}`);
    }
    const entry = responses[0];
    return [
      (entry.output as string) ?? "",
      {
        model_name: model,
        tess_response_id: entry.id as number,
        tess_root_id: entry.root_id as number,
        tess_status: entry.status as string,
        credits: (entry.credits as number) ?? 0,
        template_id: entry.template_id as number,
      },
    ];
  }

  private static _extractOutputAndMetadataFromPoll(
    data: Record<string, unknown>,
    model: string,
  ): [string, TessResponseMetadata] {
    return [
      (data.output as string) ?? "",
      {
        model_name: model,
        tess_response_id: data.id as number,
        tess_root_id: data.root_id as number,
        tess_status: data.status as string,
        credits: (data.credits as number) ?? 0,
        template_id: data.template_id as number,
      },
    ];
  }

  // ------------------------------------------------------------------
  // Retry error raising
  // ------------------------------------------------------------------

  private _raiseAfterRetries(
    lastError: Error | undefined,
    lastOutput: string,
  ): never {
    if (lastError instanceof TessAPIError) throw lastError;
    if (lastError instanceof ToolCallParseError) {
      throw new Error(
        `Failed to parse tool calls from model response after ${this.maxRetries + 1} attempts: ${lastOutput}`,
      );
    }
    if (lastError?.message === "__empty_response__") {
      throw new Error(
        `Tess API returned an empty response after ${this.maxRetries + 1} attempts`,
      );
    }
    throw new Error(
      `Tess API request failed after ${this.maxRetries + 1} attempts`,
    );
  }

  // ------------------------------------------------------------------
  // SSE parsing
  // ------------------------------------------------------------------

  private static _parseSseLine(
    line: string,
  ): Record<string, unknown> | null {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.startsWith("data:")) return null;
    const payload = trimmed.slice("data:".length).trim();
    if (payload === "[DONE]") return null;
    try {
      return JSON.parse(payload) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  // ------------------------------------------------------------------
  // Core: _generate
  // ------------------------------------------------------------------

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const boundTools = options.tools as
      | Array<Record<string, unknown>>
      | undefined;
    const toolChoice = options.tool_choice as unknown;

    const injected = ChatTessAI._injectJsonFormatPrompt(
      messages,
      boundTools,
      toolChoice,
    );
    const [converted, fileRefs] =
      ChatTessAI._convertMessagesWithFiles(injected);

    const dynamicFileIds = fileRefs.length
      ? await this._resolveFileIds(fileRefs)
      : [];
    const mergedFileIds = ChatTessAI._mergeFileIds(
      this.fileIds,
      dynamicFileIds,
    );

    let continuationRootId: number | null = null;
    let messagesToSend = converted;
    if (this.trackConversations) {
      const [foundRootId, prefixLen] = this._findConversation(converted);
      if (foundRootId != null) {
        continuationRootId = foundRootId;
        messagesToSend = converted.slice(prefixLen);
      }
    }

    let payload = this._buildPayloadFromConverted(messagesToSend, {
      stream: false,
      rootId: continuationRootId,
      fileIds: mergedFileIds,
    });

    let lastError: Error | undefined;
    let lastOutput = "";

    const discardContinuation = () => {
      if (continuationRootId == null) return;
      this._invalidateRootId(continuationRootId);
      continuationRootId = null;
      payload = this._buildPayloadFromConverted(converted, {
        stream: false,
        fileIds: mergedFileIds,
      });
    };

    const stop = options.stop as string[] | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      let output: string;
      let metadata: TessResponseMetadata;

      try {
        [output, metadata] = await this._executeRequest(payload);
      } catch (exc) {
        if (
          exc instanceof TessAuthenticationError ||
          exc instanceof TessPayloadTooLargeError
        ) {
          throw exc;
        }
        if (exc instanceof TessRateLimitError) {
          lastError = exc;
          discardContinuation();
          if (attempt < this.maxRetries) await sleep(exc.retryAfter);
          continue;
        }
        if (exc instanceof TessAPIError || exc instanceof Error) {
          lastError = exc;
          discardContinuation();
          if (attempt < this.maxRetries)
            await sleep(Math.min(2 ** attempt, 8));
          continue;
        }
        throw exc;
      }

      if (!output.trim()) {
        lastError = new Error("__empty_response__");
        lastOutput = output;
        discardContinuation();
        if (attempt < this.maxRetries)
          await sleep(Math.min(2 ** attempt, 8));
        continue;
      }

      let msg: AIMessage;
      try {
        msg = this._outputToAssistantMessage(
          output,
          boundTools,
          metadata,
          stop,
        );
      } catch (exc) {
        if (exc instanceof ToolCallParseError) {
          lastError = exc;
          lastOutput = exc.rawOutput;
          discardContinuation();
          if (attempt < this.maxRetries)
            await sleep(Math.min(2 ** attempt, 8));
          continue;
        }
        throw exc;
      }

      this._trackAfterResponse(converted, msg, metadata, output);

      const enc = this._getEncoding();
      const inputTokens = converted.reduce(
        (sum, m) => sum + enc.encode(m.content ?? "").length,
        0,
      );
      const outputTokens = enc.encode(output).length;
      msg.usage_metadata = {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      };

      return {
        generations: [
          {
            text: typeof msg.content === "string" ? msg.content : "",
            message: msg,
          },
        ],
      };
    }

    this._raiseAfterRetries(lastError, lastOutput);
  }

  // ------------------------------------------------------------------
  // Core: streaming
  // ------------------------------------------------------------------

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    // Bypass SSE streaming - Cloudflare's ~100s proxy timeout kills
    // the long-lived connection (HTTP 524) and leaves a phantom
    // server-side execution that causes 500 on subsequent requests.
    const result = await this._generate(messages, options, runManager);
    const genMsg = result.generations[0].message;
    const content = typeof genMsg.content === "string" ? genMsg.content : "";
    const chunk = new ChatGenerationChunk({
      text: content,
      message: new AIMessageChunk({
        content: genMsg.content,
        tool_calls: (genMsg as AIMessage).tool_calls ?? [],
        response_metadata: genMsg.response_metadata ?? {},
        usage_metadata: (genMsg as AIMessage).usage_metadata ?? undefined,
      }),
    });
    yield chunk;
  }
}
