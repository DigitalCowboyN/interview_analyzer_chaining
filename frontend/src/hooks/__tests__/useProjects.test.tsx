import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useProjects, useProject } from "@/hooks/useProjects";
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

describe("useProjects", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches /ui/projects and returns the projects array", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      projects: [{ project_id: "p1", interview_count: 3, kind: "real", suite: null }],
    } as never);

    const { result } = renderHook(() => useProjects(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/ui/projects");
    expect(result.current.data).toEqual([
      { project_id: "p1", interview_count: 3, kind: "real", suite: null },
    ]);
  });

  it("surfaces an empty list without erroring", async () => {
    vi.mocked(apiGet).mockResolvedValue({ projects: [] } as never);

    const { result } = renderHook(() => useProjects(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual([]);
  });
});

describe("useProject", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("returns the matching project row, or undefined when unlisted", async () => {
    vi.mocked(apiGet).mockResolvedValue({
      projects: [
        { project_id: "a", interview_count: 1, kind: "real", suite: null },
        { project_id: "b", interview_count: 2, kind: "real", suite: null },
      ],
    } as never);

    const { result } = renderHook(() => useProject("b"), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.project).toEqual({
      project_id: "b",
      interview_count: 2,
      kind: "real",
      suite: null,
    });

    const { result: missing } = renderHook(() => useProject("zzz"), { wrapper });
    await waitFor(() => expect(missing.current.isLoading).toBe(false));
    expect(missing.current.project).toBeUndefined();
  });
});
