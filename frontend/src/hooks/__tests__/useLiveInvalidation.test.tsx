import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import {
  useLiveInvalidation,
  buildStreamUrl,
  keysForSurface,
  DEFAULT_COALESCE_MS,
  DEFAULT_TRAILING_MS,
} from "@/hooks/useLiveInvalidation";
import { queryKeys } from "@/hooks/queryKeys";

/**
 * Frontend live-invalidation layer (M5.1 Task 5). The SSE contract is thin
 * surface tags only (`{surface, interview_id?, project_id?}`); the backend
 * (src/ui/notifications.py::NotificationHub._matches) only ever delivers a
 * notification whose ids match the subscription's own scope exactly, so the
 * mapping below targets the HOOK's own scopes, not fields off the message.
 */

class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  static reset() {
    MockEventSource.instances = [];
  }
}

function makeWrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return { client, Wrapper };
}

function latestSource(): MockEventSource {
  const source = MockEventSource.instances[MockEventSource.instances.length - 1];
  if (!source) throw new Error("no EventSource instance constructed");
  return source;
}

function send(surface: string, extra?: Record<string, string>) {
  latestSource().onmessage?.({ data: JSON.stringify({ surface, ...extra }) });
}

describe("useLiveInvalidation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockEventSource.reset();
    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  describe("pure helpers", () => {
    it("buildStreamUrl includes both params when both scopes are present", () => {
      expect(buildStreamUrl({ interviewId: "i1", projectId: "p1" })).toBe(
        "/api/ui/streams/events?interview_id=i1&project_id=p1",
      );
    });

    it("buildStreamUrl includes only interview_id when only interviewId is scoped", () => {
      expect(buildStreamUrl({ interviewId: "i1" })).toBe(
        "/api/ui/streams/events?interview_id=i1",
      );
    });

    it("buildStreamUrl includes only project_id when only projectId is scoped", () => {
      expect(buildStreamUrl({ projectId: "p1" })).toBe(
        "/api/ui/streams/events?project_id=p1",
      );
    });

    it("keysForSurface maps transcript -> transcript(interviewId)", () => {
      expect(keysForSurface("transcript", { interviewId: "i1", projectId: "p1" })).toEqual([
        queryKeys.transcript("i1"),
      ]);
    });

    it("keysForSurface maps interviews -> interviews(projectId)", () => {
      expect(keysForSurface("interviews", { interviewId: "i1", projectId: "p1" })).toEqual([
        queryKeys.interviews("p1"),
      ]);
    });

    it("keysForSurface maps project -> transcript(interviewId) + persons(projectId) when interviewId scope present", () => {
      expect(keysForSurface("project", { interviewId: "i1", projectId: "p1" })).toEqual([
        queryKeys.transcript("i1"),
        queryKeys.persons("p1"),
      ]);
    });

    it("keysForSurface maps project -> only persons(projectId) when no interviewId scope", () => {
      expect(keysForSurface("project", { projectId: "p1" })).toEqual([
        queryKeys.persons("p1"),
      ]);
    });

    it("keysForSurface maps resync -> all keys this hook watches", () => {
      expect(keysForSurface("resync", { interviewId: "i1", projectId: "p1" })).toEqual([
        queryKeys.transcript("i1"),
        queryKeys.interviews("p1"),
        queryKeys.persons("p1"),
      ]);
    });

    it("keysForSurface returns [] for an unknown surface", () => {
      expect(keysForSurface("bogus", { interviewId: "i1", projectId: "p1" })).toEqual([]);
    });
  });

  it("defaults coalesceMs=500 and trailingMs=2000 when timing is not provided", () => {
    expect(DEFAULT_COALESCE_MS).toBe(500);
    expect(DEFAULT_TRAILING_MS).toBe(2000);

    const { client, Wrapper } = makeWrapper();
    const spy = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useLiveInvalidation({ interviewId: "i1" }), { wrapper: Wrapper });

    send("transcript");
    expect(spy).toHaveBeenCalledTimes(1); // immediate

    // A second notification just under the default 500ms coalesce window
    // must be coalesced (no second immediate invalidate).
    vi.advanceTimersByTime(499);
    send("transcript");
    expect(spy).toHaveBeenCalledTimes(1);

    // The trailing refetch fires 2000ms after the MOST RECENT notification,
    // not the first — so advancing only 2000ms from the first notification
    // must not yet fire it (only 1501ms have passed since the 2nd notify).
    vi.advanceTimersByTime(1999);
    expect(spy).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(1);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("no scopes -> idle status, no EventSource constructed", () => {
    const { Wrapper } = makeWrapper();
    const { result } = renderHook(() => useLiveInvalidation({}), { wrapper: Wrapper });
    expect(result.current).toBe("idle");
    expect(MockEventSource.instances).toHaveLength(0);
  });

  it("onopen -> live, onerror -> offline, and a later reopen flips back to live", () => {
    const { Wrapper } = makeWrapper();
    const { result } = renderHook(() => useLiveInvalidation({ interviewId: "i1" }), {
      wrapper: Wrapper,
    });
    expect(result.current).toBe("idle");

    act(() => {
      latestSource().onopen?.();
    });
    expect(result.current).toBe("live");

    act(() => {
      latestSource().onerror?.();
    });
    expect(result.current).toBe("offline");

    act(() => {
      latestSource().onopen?.();
    });
    expect(result.current).toBe("live");
  });

  it("catches up (resync) on reconnect after going offline, but not on first connect", () => {
    // M5.1 final review, Important #2: after a browser<->backend outage the
    // browser's EventSource auto-reconnects; notifications published during
    // the gap are gone, so onopen must invalidate every watched key — else
    // the badge reads "live" over stale data. First connect must NOT.
    const { client, Wrapper } = makeWrapper();
    const spy = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useLiveInvalidation({ interviewId: "i1", projectId: "p1" }), {
      wrapper: Wrapper,
    });

    act(() => {
      latestSource().onopen?.();
    });
    expect(spy).not.toHaveBeenCalled(); // first connect: nothing missed

    act(() => {
      latestSource().onerror?.();
    });
    act(() => {
      latestSource().onopen?.();
    });

    // Reconnect catch-up invalidates every key this hook watches.
    for (const key of [
      queryKeys.transcript("i1"),
      queryKeys.interviews("p1"),
      queryKeys.persons("p1"),
    ]) {
      expect(spy).toHaveBeenCalledWith({ queryKey: key, exact: true });
    }
  });

  it("closes the EventSource on unmount", () => {
    const { Wrapper } = makeWrapper();
    const { unmount } = renderHook(() => useLiveInvalidation({ interviewId: "i1" }), {
      wrapper: Wrapper,
    });
    const source = latestSource();
    expect(source.closed).toBe(false);
    unmount();
    expect(source.closed).toBe(true);
  });

  it("ignores unparsable messages", () => {
    const { client, Wrapper } = makeWrapper();
    const spy = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useLiveInvalidation({ interviewId: "i1" }), { wrapper: Wrapper });

    latestSource().onmessage?.({ data: "not json" });
    expect(spy).not.toHaveBeenCalled();
  });

  describe("notification -> invalidation mapping (transcript page scope)", () => {
    function setup() {
      const { client, Wrapper } = makeWrapper();
      const spy = vi.spyOn(client, "invalidateQueries");
      renderHook(
        () => useLiveInvalidation({ interviewId: "i1", projectId: "p1" }, { coalesceMs: 500, trailingMs: 2000 }),
        { wrapper: Wrapper },
      );
      return { spy };
    }

    it("transcript surface invalidates transcript(interviewId) only", () => {
      const { spy } = setup();
      send("transcript", { interview_id: "i1" });
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.transcript("i1"), exact: true });
    });

    it("interviews surface invalidates interviews(projectId) only", () => {
      const { spy } = setup();
      send("interviews", { project_id: "p1" });
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.interviews("p1"), exact: true });
    });

    it("project surface invalidates BOTH transcript(interviewId) and persons(projectId) when an interviewId scope is present", () => {
      const { spy } = setup();
      send("project", { project_id: "p1" });
      expect(spy).toHaveBeenCalledTimes(2);
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.transcript("i1"), exact: true });
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.persons("p1"), exact: true });
    });

    it("resync surface invalidates every key this hook watches", () => {
      const { spy } = setup();
      send("resync");
      expect(spy).toHaveBeenCalledTimes(3);
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.transcript("i1"), exact: true });
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.interviews("p1"), exact: true });
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.persons("p1"), exact: true });
    });
  });

  it("project surface invalidates only persons(projectId) on the interview-list page scope (no interviewId)", () => {
    const { client, Wrapper } = makeWrapper();
    const spy = vi.spyOn(client, "invalidateQueries");
    renderHook(() => useLiveInvalidation({ projectId: "p1" }), { wrapper: Wrapper });

    send("project", { project_id: "p1" });
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.persons("p1"), exact: true });
  });

  describe("debounce policy — per query key, immediate + coalesce + trailing reset", () => {
    it("a burst of 5 notifications for the same key yields exactly 1 immediate + 1 trailing invalidate, with the trailing timer reset on each notification", () => {
      const { client, Wrapper } = makeWrapper();
      const spy = vi.spyOn(client, "invalidateQueries");
      renderHook(
        () => useLiveInvalidation({ interviewId: "i1" }, { coalesceMs: 500, trailingMs: 2000 }),
        { wrapper: Wrapper },
      );

      // 5 notifications, 100ms apart — all within the 500ms coalesce window
      // of the first, so only the first is an immediate invalidate.
      send("transcript");
      expect(spy).toHaveBeenCalledTimes(1);
      for (let i = 0; i < 4; i += 1) {
        vi.advanceTimersByTime(100);
        send("transcript");
      }
      expect(spy).toHaveBeenCalledTimes(1);

      // Trailing timer resets on every notification — advancing only up to
      // just before 2000ms since the LAST (5th) notification must not fire
      // it yet.
      vi.advanceTimersByTime(1999);
      expect(spy).toHaveBeenCalledTimes(1);
      vi.advanceTimersByTime(1);
      expect(spy).toHaveBeenCalledTimes(2);
      expect(spy).toHaveBeenNthCalledWith(2, {
        queryKey: queryKeys.transcript("i1"),
        exact: true,
      });

      // No further invalidation happens after the trailing fire (no leaked
      // timers still pending).
      vi.advanceTimersByTime(10_000);
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it("a notification after the coalesce window has elapsed triggers a new immediate invalidate", () => {
      const { client, Wrapper } = makeWrapper();
      const spy = vi.spyOn(client, "invalidateQueries");
      renderHook(
        () => useLiveInvalidation({ interviewId: "i1" }, { coalesceMs: 500, trailingMs: 2000 }),
        { wrapper: Wrapper },
      );

      send("transcript");
      expect(spy).toHaveBeenCalledTimes(1);

      vi.advanceTimersByTime(501);
      send("transcript");
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it("debounce is scoped per query key — a burst touching two different keys immediately-invalidates each independently", () => {
      const { client, Wrapper } = makeWrapper();
      const spy = vi.spyOn(client, "invalidateQueries");
      renderHook(
        () =>
          useLiveInvalidation(
            { interviewId: "i1", projectId: "p1" },
            { coalesceMs: 500, trailingMs: 2000 },
          ),
        { wrapper: Wrapper },
      );

      send("transcript");
      send("interviews");
      expect(spy).toHaveBeenCalledTimes(2);
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.transcript("i1"), exact: true });
      expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.interviews("p1"), exact: true });
    });
  });
});
