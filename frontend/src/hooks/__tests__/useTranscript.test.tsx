import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useTranscript } from "@/hooks/useTranscript";
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

const SAMPLE_TRANSCRIPT = {
  interview_id: "i1",
  title: "Kickoff call",
  metadata: {},
  lines: [
    {
      fragment_id: "f1",
      sequence_order: 0,
      text: "Hello there",
      speaker: { speaker_id: "s1", display_name: "Speaker A" },
      person: null,
      utterance_id: null,
      segment: null,
      entities: [],
      lens_items: [],
      edited: false,
    },
  ],
};

describe("useTranscript", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches the transcript for the given interview id", async () => {
    vi.mocked(apiGet).mockResolvedValue(SAMPLE_TRANSCRIPT as never);

    const { result } = renderHook(() => useTranscript("i1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith(
      "/ui/interviews/{interview_id}/transcript",
      { params: { interview_id: "i1" } },
    );
    expect(result.current.data).toEqual(SAMPLE_TRANSCRIPT);
  });

  it("does not fetch when interviewId is empty", () => {
    const { result } = renderHook(() => useTranscript(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
