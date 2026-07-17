import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import { runCorrectionIntent } from "@/hooks/mutations";

/**
 * The intent wrapper's four outcomes, tested against its framework-agnostic
 * core (`runCorrectionIntent`) with fake timers — no React rendering needed
 * since polling/timing is the thing under test here.
 *
 * Poll contract: every 2s, max 10 tries (M5.0 Task 5 binding spec).
 */

const QUERY_KEY = ["interviews", "i1", "transcript"];

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

describe("runCorrectionIntent", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.useFakeTimers();
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("settles when the poll predicate is satisfied within bounds", async () => {
    const queryFn = vi
      .fn()
      .mockResolvedValueOnce({ reflected: false })
      .mockResolvedValueOnce({ reflected: true });
    queryClient.setQueryDefaults(QUERY_KEY, { queryFn });
    // Seed initial data so refetchQueries has a query to act on.
    queryClient.setQueryData(QUERY_KEY, { reflected: false });

    const request = vi.fn().mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const isReflected = vi.fn((data: unknown) => (data as { reflected: boolean })?.reflected);

    const outcomePromise = runCorrectionIntent({
      queryClient,
      request,
      queryKey: QUERY_KEY,
      isReflected,
    });

    // First poll tick: not yet reflected.
    await vi.advanceTimersByTimeAsync(2000);
    // Second poll tick: reflected.
    await vi.advanceTimersByTimeAsync(2000);

    const outcome = await outcomePromise;
    expect(outcome.status).toBe("settled");
    expect(outcome.notice).toBeUndefined();
  });

  it("times out (NOT an error) after 10 bounded polls with no reflection", async () => {
    const queryFn = vi.fn().mockResolvedValue({ reflected: false });
    queryClient.setQueryDefaults(QUERY_KEY, { queryFn });
    queryClient.setQueryData(QUERY_KEY, { reflected: false });

    const request = vi.fn().mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const isReflected = vi.fn(() => false);

    const outcomePromise = runCorrectionIntent({
      queryClient,
      request,
      queryKey: QUERY_KEY,
      isReflected,
    });

    await vi.advanceTimersByTimeAsync(2000 * 10);

    const outcome = await outcomePromise;
    expect(outcome.status).toBe("timeout");
    expect(outcome.notice).toEqual({
      kind: "timeout",
      message: "Still processing — check back later.",
    });
    expect(isReflected).toHaveBeenCalledTimes(10);
  });

  it("reverts with the server's detail message on 409", async () => {
    const request = vi
      .fn()
      .mockResolvedValue(mockResponse(409, { detail: "Speaker was already renamed." }));
    const isReflected = vi.fn(() => true);

    const outcome = await runCorrectionIntent({
      queryClient,
      request,
      queryKey: QUERY_KEY,
      isReflected,
    });

    expect(outcome.status).toBe("reverted");
    expect(outcome.notice).toEqual({
      kind: "conflict",
      message: "Speaker was already renamed.",
    });
    // No polling should happen on a 409 — the request never landed.
    expect(isReflected).not.toHaveBeenCalled();
  });

  it("reverts with a notice on network failure", async () => {
    const request = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
    const isReflected = vi.fn(() => true);

    const outcome = await runCorrectionIntent({
      queryClient,
      request,
      queryKey: QUERY_KEY,
      isReflected,
    });

    expect(outcome.status).toBe("reverted");
    expect(outcome.notice).toEqual({
      kind: "network",
      message: "Could not reach the server. Your change was not saved.",
    });
    expect(isReflected).not.toHaveBeenCalled();
  });
});
