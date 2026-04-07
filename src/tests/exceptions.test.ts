import { describe, it, expect } from "vitest";
import {
  TessAPIError,
  TessAuthenticationError,
  TessValidationError,
  TessPayloadTooLargeError,
  TessRateLimitError,
  TessServerError,
  raiseForTessStatus,
} from "../exceptions.js";

describe("TessAPIError hierarchy", () => {
  it("TessAPIError has statusCode and body", () => {
    const err = new TessAPIError("test error", 500, { detail: "fail" });
    expect(err.message).toBe("test error");
    expect(err.statusCode).toBe(500);
    expect(err.body).toEqual({ detail: "fail" });
    expect(err.name).toBe("TessAPIError");
    expect(err).toBeInstanceOf(Error);
  });

  it("TessAuthenticationError defaults to 403", () => {
    const err = new TessAuthenticationError("auth failed");
    expect(err.statusCode).toBe(403);
    expect(err).toBeInstanceOf(TessAPIError);
    expect(err).toBeInstanceOf(TessAuthenticationError);
    expect(err.name).toBe("TessAuthenticationError");
  });

  it("TessValidationError defaults to 400", () => {
    const err = new TessValidationError("invalid input");
    expect(err.statusCode).toBe(400);
    expect(err).toBeInstanceOf(TessAPIError);
  });

  it("TessPayloadTooLargeError defaults to 413", () => {
    const err = new TessPayloadTooLargeError("too large");
    expect(err.statusCode).toBe(413);
    expect(err).toBeInstanceOf(TessAPIError);
  });

  it("TessRateLimitError defaults to 429 with retryAfter=60", () => {
    const err = new TessRateLimitError("rate limited");
    expect(err.statusCode).toBe(429);
    expect(err.retryAfter).toBe(60);
    expect(err).toBeInstanceOf(TessAPIError);
  });

  it("TessRateLimitError accepts custom retryAfter", () => {
    const err = new TessRateLimitError("rate limited", 429, {}, 30);
    expect(err.retryAfter).toBe(30);
  });

  it("TessServerError defaults to 500", () => {
    const err = new TessServerError("server error");
    expect(err.statusCode).toBe(500);
    expect(err).toBeInstanceOf(TessAPIError);
  });
});

describe("raiseForTessStatus", () => {
  it("does nothing for status < 400", () => {
    expect(() => raiseForTessStatus(200)).not.toThrow();
    expect(() => raiseForTessStatus(301)).not.toThrow();
  });

  it("throws TessValidationError for 400", () => {
    expect(() =>
      raiseForTessStatus(400, { error: "bad request" }),
    ).toThrow(TessValidationError);
  });

  it("throws TessAuthenticationError for 403", () => {
    expect(() =>
      raiseForTessStatus(403, { error: "invalid key" }),
    ).toThrow(TessAuthenticationError);
  });

  it("throws TessPayloadTooLargeError for 413", () => {
    expect(() => raiseForTessStatus(413)).toThrow(
      TessPayloadTooLargeError,
    );
  });

  it("throws TessRateLimitError for 429 with retryAfter", () => {
    try {
      raiseForTessStatus(429, { retry_after: 30, error: "rate limited" });
      expect.fail("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(TessRateLimitError);
      expect((err as TessRateLimitError).retryAfter).toBe(30);
    }
  });

  it("throws TessRateLimitError with default retryAfter=60", () => {
    try {
      raiseForTessStatus(429, {});
      expect.fail("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(TessRateLimitError);
      expect((err as TessRateLimitError).retryAfter).toBe(60);
    }
  });

  it("throws TessServerError for 500", () => {
    expect(() => raiseForTessStatus(500)).toThrow(TessServerError);
  });

  it("throws TessServerError for any 5xx", () => {
    expect(() => raiseForTessStatus(502)).toThrow(TessServerError);
    expect(() => raiseForTessStatus(503)).toThrow(TessServerError);
  });

  it("throws TessAPIError for unknown 4xx", () => {
    expect(() => raiseForTessStatus(418)).toThrow(TessAPIError);
  });

  it("uses error message from body", () => {
    try {
      raiseForTessStatus(400, { error: "custom error message" });
      expect.fail("should have thrown");
    } catch (err) {
      expect((err as TessAPIError).message).toBe("custom error message");
    }
  });

  it("generates default error message when body has no error field", () => {
    try {
      raiseForTessStatus(500);
      expect.fail("should have thrown");
    } catch (err) {
      expect((err as TessAPIError).message).toBe(
        "Tess API error (HTTP 500)",
      );
    }
  });
});
