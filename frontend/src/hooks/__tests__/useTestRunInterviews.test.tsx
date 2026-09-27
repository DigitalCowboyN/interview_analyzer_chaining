import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useTestRunInterviews } from "@/hooks/useTestRunInterviews";
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

describe("useTestRunInterviews", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches all test-run interviews", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      interviews: [
        {
          interview_id: "a",
          project_id: "smoke-1",
          suite: "smoke",
          title: "Smoke run",
          created_at: "2026-09-27T10:00:00Z",
          fragment_count: 5,
          participants: [],
          insight_counts: {},
        },
      ],
    } as never);

    const { result } = renderHook(() => useTestRunInterviews(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/test-runs/interviews");
    expect(result.current.data).toEqual([
      {
        interview_id: "a",
        project_id: "smoke-1",
        suite: "smoke",
        title: "Smoke run",
        created_at: "2026-09-27T10:00:00Z",
        fragment_count: 5,
        participants: [],
        insight_counts: {},
      },
    ]);
  });
});
