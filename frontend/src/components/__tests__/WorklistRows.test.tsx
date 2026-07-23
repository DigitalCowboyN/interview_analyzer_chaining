import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { WorklistRows } from "@/components/WorklistRows";
import { queryKeys } from "@/hooks/queryKeys";
import { apiFetch } from "@/api/client";
import type { WorklistData } from "@/hooks/useWorklist";

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return { ...actual, apiFetch: vi.fn() };
});

const POLL_OPTIONS = { pollIntervalMs: 10, maxPollAttempts: 10 };
const WORKLIST_QUERY_KEY = queryKeys.worklist("proj1");

function mockAccepted(): Response {
  return {
    ok: true,
    status: 202,
    json: async () => ({ status: "accepted" }),
  } as Response;
}

const BASE_DATA: WorklistData = {
  lens_items: [
    {
      interview_id: "i1",
      item_id: "li1",
      node_type: "claim",
      lens: "values",
      confidence: 0.42,
      reason: "low_confidence",
    },
  ],
  claims: [
    {
      interview_id: "i2",
      claim_id: "c1",
      text: "We ship weekly.",
      kind: "assertion",
      confidence: 0.55,
      reason: "low_confidence",
    },
  ],
  entity_merge_suggestions: [
    {
      surviving_canonical_id: "cn1",
      merged_canonical_id: "cn2",
      surfaces_a: ["Acme Corp"],
      surfaces_b: ["Acme", "Acme Inc"],
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

/**
 * Renders WorklistRows behind a real QueryClient so the internal accept
 * intents (keyed to the worklist query) can poll/settle deterministically —
 * mirrors PersonPicker.test.tsx's controllable-queryFn pattern.
 */
function renderRows(data: WorklistData, mutableData?: { current: WorklistData }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const ref = mutableData ?? { current: data };
  client.setQueryDefaults(WORKLIST_QUERY_KEY, { queryFn: async () => ref.current });
  client.setQueryData(WORKLIST_QUERY_KEY, ref.current);
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  const utils = render(
    <WorklistRows projectId="proj1" data={data} pollOptions={POLL_OPTIONS} />,
    { wrapper: Wrapper },
  );
  return { ...utils, ref };
}

describe("WorklistRows", () => {
  beforeEach(() => {
    vi.mocked(apiFetch).mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a low-confidence lens item row linking into the workbench transcript", () => {
    renderRows(BASE_DATA);
    expect(screen.getByText(/values/)).toBeInTheDocument();
    expect(screen.getAllByText(/low confidence/).length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /values/ })).toHaveAttribute(
      "href",
      "/workbench/proj1/i1",
    );
  });

  it("renders a claim row linking into the workbench transcript at that interview", () => {
    renderRows(BASE_DATA);
    expect(screen.getByText("We ship weekly.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /We ship weekly\./ })).toHaveAttribute(
      "href",
      "/workbench/proj1/i2",
    );
  });

  it("renders an entity-merge suggestion with surfaces, score, and band", () => {
    renderRows(BASE_DATA);
    expect(screen.getByText(/Acme Corp/)).toBeInTheDocument();
    expect(screen.getByText(/Acme, Acme Inc/)).toBeInTheDocument();
    expect(screen.getByText(/0\.91/)).toBeInTheDocument();
    expect(screen.getByText("suggest")).toBeInTheDocument();
  });

  it("renders a person-link suggestion", () => {
    renderRows(BASE_DATA);
    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText(/Speaker A/)).toBeInTheDocument();
    expect(screen.getByText(/name_match/)).toBeInTheDocument();
  });

  it("renders the degradation banner with the exact copy while deterministic rows still render", () => {
    renderRows({ ...BASE_DATA, flags: ["embedding_unavailable"] });
    expect(
      screen.getByText("suggestions degraded — embedding provider unavailable"),
    ).toBeInTheDocument();
    // Deterministic rows (lens items, claims, person-link suggestions) still present.
    expect(screen.getByText(/values/)).toBeInTheDocument();
    expect(screen.getByText("We ship weekly.")).toBeInTheDocument();
    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
  });

  it("does not render the degradation banner when the flag is absent", () => {
    renderRows(BASE_DATA);
    expect(
      screen.queryByText("suggestions degraded — embedding provider unavailable"),
    ).not.toBeInTheDocument();
  });

  it("accepting an entity-merge suggestion calls the merge endpoint with the correct body, and the row disappears once settled", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue(mockAccepted());
    const { ref } = renderRows(BASE_DATA);

    await user.click(screen.getByRole("button", { name: "Accept merge" }));

    expect(apiFetch).toHaveBeenCalledWith("/resolution/proj1/entities/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        surviving_canonical_id: "cn1",
        merged_canonical_id: "cn2",
      }),
    });

    // Simulate the backend having caught up: the next refetch no longer
    // contains this suggestion.
    ref.current = { ...BASE_DATA, entity_merge_suggestions: [] };
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Accept merge" })).not.toBeInTheDocument(),
    );
  });

  it("accepting a person-link suggestion calls the link endpoint with display_name, and the row disappears once settled", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue(mockAccepted());
    const { ref } = renderRows(BASE_DATA);

    await user.click(screen.getByRole("button", { name: "Accept link" }));

    expect(apiFetch).toHaveBeenCalledWith("/resolution/proj1/persons/p1/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        interview_id: "i1",
        speaker_id: "s1",
        display_name: "Jane Doe",
      }),
    });

    ref.current = { ...BASE_DATA, person_link_suggestions: [] };
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Accept link" })).not.toBeInTheDocument(),
    );
  });
});
