import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useInterviews } from "@/hooks/useInterviews";
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

describe("useInterviews", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches interviews scoped to the project id", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      interviews: [
        {
          interview_id: "i1",
          title: "Kickoff",
          created_at: "2026-01-01T00:00:00Z",
          fragment_count: 12,
        },
      ],
    } as never);

    const { result } = renderHook(() => useInterviews("p1"), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith(
      "/ui/projects/{project_id}/interviews",
      { params: { project_id: "p1" } },
    );
    expect(result.current.data).toEqual([
      {
        interview_id: "i1",
        title: "Kickoff",
        created_at: "2026-01-01T00:00:00Z",
        fragment_count: 12,
      },
    ]);
  });

  it("does not fetch when projectId is empty", () => {
    const { result } = renderHook(() => useInterviews(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
