import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { apiFetch, apiGet, ApiError } from "@/api/client";

describe("apiFetch", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("injects X-User-ID from the identity store on every request, including GETs", async () => {
    window.localStorage.setItem("interview-analyzer:user-id", "alice");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch("/ui/projects", { method: "GET" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/ui/projects");
    const headers = new Headers(init.headers);
    expect(headers.get("X-User-ID")).toBe("alice");
  });

  it("defaults X-User-ID to 'dev' when no identity has been set", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await apiFetch("/ui/projects");

    const [, init] = fetchMock.mock.calls[0];
    const headers = new Headers(init.headers);
    expect(headers.get("X-User-ID")).toBe("dev");
  });

  it("apiGet resolves parsed JSON on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ projects: [] }), { status: 200 }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const result = await apiGet("/ui/projects");
    expect(result).toEqual({ projects: [] });
  });

  it("apiGet throws ApiError with server detail on non-2xx", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "not found" }), { status: 404 }),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    await expect(apiGet("/ui/projects")).rejects.toMatchObject({
      status: 404,
      detail: "not found",
    } satisfies Partial<ApiError>);
  });
});
