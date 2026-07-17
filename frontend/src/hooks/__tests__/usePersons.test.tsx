import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { usePersons, usePersonId } from "@/hooks/usePersons";
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

describe("usePersons", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("lists the project's persons", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      persons: [
        { person_id: "p1", display_name: "Jane Doe", speaker_count: 2, interview_count: 1 },
      ],
    } as never);

    const { result } = renderHook(() => usePersons("proj1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/projects/{project_id}/persons", {
      params: { project_id: "proj1" },
    });
    expect(result.current.data).toEqual([
      { person_id: "p1", display_name: "Jane Doe", speaker_count: 2, interview_count: 1 },
    ]);
  });

  it("does not fetch when projectId is empty", () => {
    const { result } = renderHook(() => usePersons(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});

describe("usePersonId (create-new derivation)", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("calls the server-side derivation endpoint and returns its person_id — never derives locally", async () => {
    vi.mocked(apiGet).mockResolvedValue({ person_id: "person-abc123" } as never);

    const { result } = renderHook(() => usePersonId());
    const personId = await result.current.derivePersonId("proj1", "Jane Doe");

    expect(apiGet).toHaveBeenCalledWith("/ui/projects/{project_id}/person-id", {
      params: { project_id: "proj1" },
      query: { display_name: "Jane Doe" },
    });
    expect(personId).toBe("person-abc123");
  });
});
