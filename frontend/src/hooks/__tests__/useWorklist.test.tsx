import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWorklist } from "@/hooks/useWorklist";
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

const RAW_RESPONSE = {
  lens_items: [
    {
      interview_id: "i1",
      item_id: "li1",
      node_type: "claim",
      lens: "values",
      confidence: 0.4,
      reason: "low_confidence",
    },
  ],
  claims: [
    {
      interview_id: "i1",
      claim_id: "c1",
      text: "We ship weekly.",
      kind: "assertion",
      confidence: 0.5,
      reason: "low_confidence",
    },
  ],
  entity_merge_suggestions: [
    {
      surviving_canonical_id: "cn1",
      merged_canonical_id: "cn2",
      surfaces_a: ["Acme Corp"],
      surfaces_b: ["Acme"],
      score: 0.91,
      band: "suggest",
    },
  ],
  person_link_suggestions: [
    {
      person_id: "p1",
      display_name: "Jane Doe",
      interview_id: "i1",
      speaker_id: "s1",
      speaker_display_name: "Speaker A",
      reason: "name_match",
    },
  ],
  flags: [],
};

describe("useWorklist", () => {
  beforeEach(() => {
    vi.mocked(apiGet).mockReset();
  });

  it("fetches the project's review worklist", async () => {
    vi.mocked(apiGet).mockResolvedValue(RAW_RESPONSE as never);

    const { result } = renderHook(() => useWorklist("proj1"), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(apiGet).toHaveBeenCalledWith("/review/worklist", {
      query: { project_id: "proj1" },
    });
    expect(result.current.data).toEqual(RAW_RESPONSE);
  });

  it("does not fetch when projectId is empty", () => {
    const { result } = renderHook(() => useWorklist(""), { wrapper });
    expect(result.current.fetchStatus).toBe("idle");
    expect(apiGet).not.toHaveBeenCalled();
  });
});
