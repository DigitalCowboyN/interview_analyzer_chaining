import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useInsights } from "@/hooks/useInsights";
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

const MEETING_ITEM = {
  item_id: "d1",
  node_type: "Decision",
  confidence: 0.9,
  locked: false,
  supporting_fragment_ids: ["f1"],
  fields: { text: "Ship CSV export" },
};

const PERSONA_ITEM = {
  item_id: "g1",
  node_type: "Goal",
  confidence: 0.8,
  locked: true,
  supporting_fragment_ids: ["f2"],
  fields: { text: "Understand drill-down pain" },
};

describe("useInsights", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("returns one insight per lens, with lens set and text from fields.text, calling both lenses with limit 500", async () => {
    vi.mocked(apiGet).mockImplementation(async (_path, options) => {
      const lens = options?.params?.lens;
      if (lens === "meeting_minutes") return { items: [MEETING_ITEM] } as never;
      if (lens === "persona") return { items: [PERSONA_ITEM] } as never;
      throw new Error(`unexpected lens ${String(lens)}`);
    });

    const { result } = renderHook(() => useInsights("i1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual([
      {
        item_id: "d1",
        node_type: "Decision",
        lens: "meeting_minutes",
        text: "Ship CSV export",
        confidence: 0.9,
        locked: false,
        supporting_fragment_ids: ["f1"],
      },
      {
        item_id: "g1",
        node_type: "Goal",
        lens: "persona",
        text: "Understand drill-down pain",
        confidence: 0.8,
        locked: true,
        supporting_fragment_ids: ["f2"],
      },
    ]);

    expect(apiGet).toHaveBeenCalledWith(
      "/interviews/{interview_id}/lenses/{lens}/items",
      { params: { interview_id: "i1", lens: "meeting_minutes" }, query: { limit: 500 } },
    );
    expect(apiGet).toHaveBeenCalledWith(
      "/interviews/{interview_id}/lenses/{lens}/items",
      { params: { interview_id: "i1", lens: "persona" }, query: { limit: 500 } },
    );
  });

  it("keeps the fulfilled lens's items when the other lens rejects", async () => {
    vi.mocked(apiGet).mockImplementation(async (_path, options) => {
      const lens = options?.params?.lens;
      if (lens === "meeting_minutes") return { items: [MEETING_ITEM] } as never;
      throw new Error("persona lens down");
    });

    const { result } = renderHook(() => useInsights("i1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual([
      {
        item_id: "d1",
        node_type: "Decision",
        lens: "meeting_minutes",
        text: "Ship CSV export",
        confidence: 0.9,
        locked: false,
        supporting_fragment_ids: ["f1"],
      },
    ]);
  });

  it("errors when both lenses reject", async () => {
    vi.mocked(apiGet).mockRejectedValue(new Error("down"));

    const { result } = renderHook(() => useInsights("i1"), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});
