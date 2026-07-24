import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { usePersonDetail } from "@/hooks/usePersonDetail";
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

const SAMPLE_PERSON = {
  person_id: "p1",
  display_name: "Jane Doe",
  links: [
    {
      interview_id: "i1",
      interview_title: "Kickoff call",
      speaker_id: "s1",
      speaker_display_name: "Speaker A",
    },
  ],
  contributes_to_persona: true,
};

describe("usePersonDetail", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches the person core view for the given project/person", async () => {
    vi.mocked(apiGet).mockResolvedValue(SAMPLE_PERSON as never);

    const { result } = renderHook(() => usePersonDetail("proj1", "p1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/persons/{project_id}/{person_id}", {
      params: { project_id: "proj1", person_id: "p1" },
    });
    expect(result.current.data).toEqual(SAMPLE_PERSON);
  });

  it("does not fetch when either id is empty", () => {
    const { result } = renderHook(() => usePersonDetail("", "p1"), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
