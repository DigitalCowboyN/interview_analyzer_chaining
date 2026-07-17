import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { usePersonaCards } from "@/hooks/usePersonaCards";
import { apiGet } from "@/api/client";

vi.mock("@/api/client", () => ({
  apiGet: vi.fn(),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("usePersonaCards", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("lists the project's persona cards", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      personas: [
        {
          person_id: "p1",
          display_name: "Jane Doe",
          trait_count: 2,
          goal_count: 1,
          pain_point_count: 1,
          quote_count: 3,
          representative_quote: "I just want it to work.",
          interview_ids: ["i1", "i2"],
        },
      ],
    } as never);

    const { result } = renderHook(() => usePersonaCards("proj1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/projects/{project_id}/personas", {
      params: { project_id: "proj1" },
    });
    expect(result.current.data?.[0].display_name).toBe("Jane Doe");
    expect(result.current.data?.[0].interview_ids).toEqual(["i1", "i2"]);
  });

  it("does not fetch when projectId is empty", () => {
    const { result } = renderHook(() => usePersonaCards(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
