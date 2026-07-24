import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { usePersonaDetail } from "@/hooks/usePersonaDetail";
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

const SAMPLE_PERSONA = {
  person_id: "p1",
  display_name: "Jane Doe",
  dimensions: {
    traits: [
      {
        item_id: "t1",
        text: "Detail-oriented",
        confidence: 0.9,
        interview_id: "i1",
        interview_title: "Kickoff call",
      },
    ],
    goals: [],
    pain_points: [],
    notable_quotes: [],
  },
};

describe("usePersonaDetail", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches the persona core view for the given project/person", async () => {
    vi.mocked(apiGet).mockResolvedValue(SAMPLE_PERSONA as never);

    const { result } = renderHook(() => usePersonaDetail("proj1", "p1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/personas/{project_id}/{person_id}", {
      params: { project_id: "proj1", person_id: "p1" },
    });
    expect(result.current.data).toEqual(SAMPLE_PERSONA);
  });

  it("does not fetch when either id is empty", () => {
    const { result } = renderHook(() => usePersonaDetail("proj1", ""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
