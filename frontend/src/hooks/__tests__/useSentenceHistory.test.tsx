import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { apiGet } from "@/api/client";

vi.mock("@/api/client", () => ({
  apiGet: vi.fn(),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useSentenceHistory", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches history keyed by interview id and sequence_order (sentence_index) when enabled", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      sentence_id: "sentence-uuid",
      interview_id: "i1",
      sentence_index: 3,
      current_version: 2,
      current_text: "Edited text",
      is_edited: true,
      event_count: 2,
      events: [],
    } as never);

    const { result } = renderHook(() => useSentenceHistory("i1", 3, true), {
      wrapper,
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith(
      "/edits/sentences/{interview_id}/{sentence_index}/history",
      { params: { interview_id: "i1", sentence_index: 3 } },
    );
  });

  it("does not fetch when disabled (lazy — only fires once the panel opens)", () => {
    const { result } = renderHook(() => useSentenceHistory("i1", 3, false), {
      wrapper,
    });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
