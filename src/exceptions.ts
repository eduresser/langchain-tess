/**
 * Typed exception hierarchy for the Tess AI API.
 *
 * Maps HTTP status codes returned by the Tess API to specific error
 * classes so callers can handle authentication failures, rate limits, and
 * validation errors differently from generic server errors.
 */

export class TessAPIError extends Error {
  statusCode: number;
  body?: Record<string, unknown>;

  constructor(
    message: string,
    statusCode: number,
    body?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "TessAPIError";
    this.statusCode = statusCode;
    this.body = body;
  }
}

/** Raised on HTTP 403 - invalid or expired API key. */
export class TessAuthenticationError extends TessAPIError {
  constructor(
    message: string,
    statusCode: number = 403,
    body?: Record<string, unknown>,
  ) {
    super(message, statusCode, body);
    this.name = "TessAuthenticationError";
  }
}

/** Raised on HTTP 400 - invalid request payload. */
export class TessValidationError extends TessAPIError {
  constructor(
    message: string,
    statusCode: number = 400,
    body?: Record<string, unknown>,
  ) {
    super(message, statusCode, body);
    this.name = "TessValidationError";
  }
}

/** Raised on HTTP 413 - request body exceeds allowed size. */
export class TessPayloadTooLargeError extends TessAPIError {
  constructor(
    message: string,
    statusCode: number = 413,
    body?: Record<string, unknown>,
  ) {
    super(message, statusCode, body);
    this.name = "TessPayloadTooLargeError";
  }
}

/** Raised on HTTP 429 - rate limit exceeded. */
export class TessRateLimitError extends TessAPIError {
  retryAfter: number;

  constructor(
    message: string,
    statusCode: number = 429,
    body?: Record<string, unknown>,
    retryAfter: number = 60,
  ) {
    super(message, statusCode, body);
    this.name = "TessRateLimitError";
    this.retryAfter = retryAfter;
  }
}

/** Raised on HTTP 5xx - internal server error. */
export class TessServerError extends TessAPIError {
  constructor(
    message: string,
    statusCode: number = 500,
    body?: Record<string, unknown>,
  ) {
    super(message, statusCode, body);
    this.name = "TessServerError";
  }
}

const STATUS_TO_EXCEPTION: Record<
  number,
  new (
    message: string,
    statusCode: number,
    body?: Record<string, unknown>,
  ) => TessAPIError
> = {
  400: TessValidationError,
  403: TessAuthenticationError,
  413: TessPayloadTooLargeError,
  429: TessRateLimitError,
  500: TessServerError,
};

/**
 * Raise the appropriate TessAPIError subclass for the given status code.
 * Does nothing when statusCode < 400. For 429 responses, extracts
 * retryAfter from the body when available.
 */
export function raiseForTessStatus(
  statusCode: number,
  body?: Record<string, unknown>,
): void {
  if (statusCode < 400) return;

  const safeBody = body ?? {};
  const errorMsg =
    (safeBody.error as string) ?? `Tess API error (HTTP ${statusCode})`;

  let ExcCls = STATUS_TO_EXCEPTION[statusCode];
  if (!ExcCls) {
    ExcCls = statusCode >= 500 ? TessServerError : TessAPIError;
  }

  if (ExcCls === TessRateLimitError) {
    const retryAfter = Number(safeBody.retry_after ?? 60);
    throw new TessRateLimitError(errorMsg, statusCode, safeBody, retryAfter);
  }

  throw new ExcCls(errorMsg, statusCode, safeBody);
}
