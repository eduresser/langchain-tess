import type {
  BaseChatModelCallOptions,
  ToolChoice,
} from "@langchain/core/language_models/chat_models";

export interface ChatTessAIInput {
  /** Tess AI API key. Can also be set via TESS_API_KEY env var. */
  apiKey?: string;
  /** The Tess AI agent ID to use. */
  agentId: number;
  /** Model name to use. */
  model?: string;
  /** Temperature for generation (0.0 - 1.0). */
  temperature?: number;
  /** Tess tools: 'no-tools', 'internet', 'twitter', 'wikipedia', etc. */
  tools?: string;
  /** Tess workspace ID. */
  workspaceId: number;
  /** API base URL. */
  baseUrl?: string;
  /** Request timeout in seconds. */
  timeout?: number;
  /** Max retry attempts. */
  maxRetries?: number;
  /** Wait for the execution to complete (API has a 100s timeout). */
  waitExecution?: boolean;
  /** Seconds between polling requests. */
  pollingInterval?: number;
  /** File IDs to attach to every execution. */
  fileIds?: number[];
  /** Automatically reuse root_id to continue Tess conversations. */
  trackConversations?: boolean;
  /** Maximum number of conversations kept in the LRU cache. */
  maxTrackedConversations?: number;
}

export interface ChatTessAICallOptions extends BaseChatModelCallOptions {
  tools?: Array<Record<string, unknown>>;
  tool_choice?: ToolChoice;
}

export interface TessMessage {
  role: string;
  content: string;
}

export interface TessResponseMetadata {
  model_name?: string;
  tess_response_id?: number;
  tess_root_id?: number;
  tess_status?: string;
  credits?: number;
  template_id?: number;
  tess_parse_degraded?: boolean;
}

export interface FileRef {
  mimeType: string;
  data?: Uint8Array;
  url?: string;
  fileId?: number;
  messageIndex: number;
}

export interface ParsedToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string;
}
