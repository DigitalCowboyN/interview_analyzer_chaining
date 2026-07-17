import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import {
  useTextEditIntent,
  useSpeakerRenameIntent,
  useFragmentReattributeIntent,
  useSegmentRemoveIntent,
  useLensItemOverrideIntent,
} from "@/hooks/mutations";
import { apiFetch } from "@/api/client";

/**
 * One test per correction flow (M5.0 Task 5): asserts the exact endpoint and
 * body the flow hook sends via `apiFetch`. Timing/outcome behavior for the
 * shared wrapper itself (settle/timeout/409/network) is covered by
 * mutations.test.ts against the framework-agnostic core with fake timers.
 * These tests use real timers (the poll predicate is satisfied on the very
 * first tick, so each test waits out one real poll interval) — mixing fake
 * timers with RTL's `waitFor` (which itself polls via setTimeout) is
 * unreliable, so real timers keep these deterministic. Each hook is given a
 * small injected poll interval (POLL_OPTIONS below) so the suite doesn't pay
 * the production 2s cost per test; the default (2s x 10 tries) is asserted
 * separately in mutations.test.ts.
 */

const POLL_OPTIONS = { pollIntervalMs: 10, maxPollAttempts: 10 };

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return { ...actual, apiFetch: vi.fn() };
});

const QUERY_KEY = ["interviews", "i1", "transcript"] as const;

function mockResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function makeWrapper(initialData: unknown) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryDefaults(QUERY_KEY, { queryFn: async () => initialData });
  client.setQueryData(QUERY_KEY, initialData);
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return Wrapper;
}

describe("correction flow hooks — endpoint/body pinning", () => {
  beforeEach(() => {
    vi.mocked(apiFetch).mockReset();
  });

  it("text edit: POST /edits/sentences/{interview_id}/{sentence_index}/edit", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [{ fragment_id: "f1", sequence_order: 3, edited: true, speaker: null, segment: null, lens_items: [] }],
    });

    const { result } = renderHook(() => useTextEditIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.editText(3, "corrected text", "typo fix");

    expect(apiFetch).toHaveBeenCalledWith("/edits/sentences/i1/3/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "corrected text", editor_type: "human", note: "typo fix" }),
    });

    const outcome = await outcomePromise;
    expect(outcome).toMatchObject({ status: "settled" });
  });

  it("text edit omits note when not provided", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [{ fragment_id: "f1", sequence_order: 0, edited: true, speaker: null, segment: null, lens_items: [] }],
    });

    const { result } = renderHook(() => useTextEditIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.editText(0, "new text");

    expect(apiFetch).toHaveBeenCalledWith("/edits/sentences/i1/0/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "new text", editor_type: "human" }),
    });

    await outcomePromise;
  });

  it("speaker rename: POST /speakers/{interview_id}/{speaker_id}/rename", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [
        {
          fragment_id: "f1",
          sequence_order: 0,
          edited: false,
          speaker: { speaker_id: "s1", display_name: "Jane Doe" },
          segment: null,
          lens_items: [],
        },
      ],
    });

    const { result } = renderHook(() => useSpeakerRenameIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.renameSpeaker("s1", "Jane Doe");

    expect(apiFetch).toHaveBeenCalledWith("/speakers/i1/s1/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_display_name: "Jane Doe" }),
    });

    const outcome = await outcomePromise;
    expect(outcome).toMatchObject({ status: "settled" });
  });

  it("fragment reattribute: POST /speakers/{interview_id}/fragments/{index}/reattribute", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [
        {
          fragment_id: "f1",
          sequence_order: 4,
          edited: false,
          speaker: { speaker_id: "s2", display_name: "Speaker B" },
          segment: null,
          lens_items: [],
        },
      ],
    });

    const { result } = renderHook(() => useFragmentReattributeIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.reattributeFragment(4, "s2");

    expect(apiFetch).toHaveBeenCalledWith("/speakers/i1/fragments/4/reattribute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_speaker_id: "s2" }),
    });

    const outcome = await outcomePromise;
    expect(outcome).toMatchObject({ status: "settled" });
  });

  it("segment remove: DELETE /segments/{interview_id}/{segment_id}?reason=", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    // Reflected = segment no longer present on any line.
    const wrapper = makeWrapper({
      lines: [{ fragment_id: "f1", sequence_order: 0, edited: false, speaker: null, segment: null, lens_items: [] }],
    });

    const { result } = renderHook(() => useSegmentRemoveIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.removeSegment("seg1", "wrong boundary");

    expect(apiFetch).toHaveBeenCalledWith(
      "/segments/i1/seg1?reason=wrong%20boundary",
      { method: "DELETE" },
    );

    const outcome = await outcomePromise;
    expect(outcome).toMatchObject({ status: "settled" });
  });

  it("segment remove without a reason omits the query string", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [{ fragment_id: "f1", sequence_order: 0, edited: false, speaker: null, segment: null, lens_items: [] }],
    });

    const { result } = renderHook(() => useSegmentRemoveIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.removeSegment("seg1");

    expect(apiFetch).toHaveBeenCalledWith("/segments/i1/seg1", { method: "DELETE" });

    await outcomePromise;
  });

  it("lens-item override: POST /lenses/{interview_id}/items/{item_id}/override", async () => {
    vi.mocked(apiFetch).mockResolvedValue(mockResponse(202, { status: "accepted" }));
    const wrapper = makeWrapper({
      lines: [
        {
          fragment_id: "f1",
          sequence_order: 0,
          edited: false,
          speaker: null,
          segment: null,
          lens_items: [{ item_id: "li1", human_locked: true }],
        },
      ],
    });

    const { result } = renderHook(() => useLensItemOverrideIntent("i1", QUERY_KEY, POLL_OPTIONS), { wrapper });
    const outcomePromise = result.current.overrideLensItem(
      "li1",
      { text: "Ships faster" },
      "confidence too low",
    );

    expect(apiFetch).toHaveBeenCalledWith("/lenses/i1/items/li1/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fields_overridden: { text: "Ships faster" },
        note: "confidence too low",
      }),
    });

    const outcome = await outcomePromise;
    expect(outcome).toMatchObject({ status: "settled" });
  });
});
