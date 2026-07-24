import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { LineDetailPanel } from "@/components/LineDetailPanel";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { usePersons, usePersonId } from "@/hooks/usePersons";
import type { TranscriptLineData } from "@/hooks/useTranscript";
import { apiFetch } from "@/api/client";

vi.mock("@/hooks/useSentenceHistory", () => ({
  useSentenceHistory: vi.fn(),
}));

vi.mock("@/hooks/usePersons", () => ({
  usePersons: vi.fn(),
  usePersonId: vi.fn(),
}));

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return { ...actual, apiFetch: vi.fn() };
});

const LINE: TranscriptLineData = {
  fragment_id: "f1",
  sequence_order: 5,
  text: "Full sentence text",
  speaker: { speaker_id: "s1", display_name: "Speaker A" },
  person: null,
  utterance_id: null,
  segment: { segment_id: "seg1", topic: "Onboarding" },
  entities: [{ surface: "Acme Corp", entity_type: "ORG" }],
  lens_items: [
    {
      item_id: "li1",
      lens: "persona",
      node_type: "Goal",
      text: "Ship faster",
      confidence: 0.87,
      human_locked: true,
    },
  ],
  edited: false,
};

function renderPanel(line: TranscriptLineData = LINE, onClose = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return render(
    <LineDetailPanel projectId="p1" interviewId="i1" line={line} onClose={onClose} />,
    { wrapper: Wrapper },
  );
}

describe("LineDetailPanel", () => {
  beforeEach(() => {
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);
    vi.mocked(apiFetch).mockReset();
    vi.mocked(usePersons).mockReturnValue({
      data: [{ person_id: "p1", display_name: "Jane Doe", speaker_count: 1, interview_count: 1 }],
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    vi.mocked(usePersonId).mockReturnValue({
      derivePersonId: vi.fn().mockResolvedValue("person-new"),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders full text, entities, and lens items (lens, type, text, confidence, lock state)", () => {
    renderPanel();

    expect(screen.getByText("Full sentence text")).toBeInTheDocument();
    expect(screen.getByText("Acme Corp")).toBeInTheDocument();
    expect(screen.getByText("(ORG)")).toBeInTheDocument();

    expect(screen.getByText("persona")).toBeInTheDocument();
    expect(screen.getByText("Goal")).toBeInTheDocument();
    expect(screen.getByText("Ship faster")).toBeInTheDocument();
    expect(screen.getByText("87%")).toBeInTheDocument();
    expect(screen.getByText("locked")).toBeInTheDocument();
  });

  it("triggers the lazy history fetch (hook called with sequence_order as sentence_index, enabled=true)", () => {
    renderPanel();
    expect(useSentenceHistory).toHaveBeenCalledWith("i1", 5, true);
  });

  it("renders history events once loaded", () => {
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: {
        sentence_id: "sid",
        interview_id: "i1",
        sentence_index: 5,
        current_version: 2,
        current_text: "Full sentence text",
        is_edited: true,
        event_count: 1,
        events: [
          {
            event_type: "SentenceEdited",
            version: 2,
            occurred_at: "2026-01-02T00:00:00Z",
            actor: { actor_type: "human", user_id: "dev" },
            correlation_id: "corr-1",
            data: {},
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPanel();
    expect(screen.getByText("SentenceEdited")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    const onClose = vi.fn();
    renderPanel(LINE, onClose);
    screen.getByRole("button", { name: "Close detail panel" }).click();
    expect(onClose).toHaveBeenCalled();
  });

  // --- Correction affordances (M5.0 Task 5) ---

  it("text edit: opening the editor and saving calls editText's endpoint/body", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    await user.click(screen.getByRole("button", { name: "Edit text" }));
    const textarea = screen.getByLabelText("Edit sentence text");
    await user.clear(textarea);
    await user.type(textarea, "corrected text");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(apiFetch).toHaveBeenCalledWith("/edits/sentences/i1/5/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "corrected text", editor_type: "human" }),
    });
  });

  it("speaker rename: opening the editor and saving calls renameSpeaker's endpoint/body", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    await user.click(screen.getByRole("button", { name: "Rename speaker" }));
    const input = screen.getByLabelText("New speaker display name");
    await user.clear(input);
    await user.type(input, "Jane Doe");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(apiFetch).toHaveBeenCalledWith("/speakers/i1/s1/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_display_name: "Jane Doe" }),
    });
  });

  it("fragment reattribute: entering a target speaker id calls reattributeFragment's endpoint/body", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    await user.click(screen.getByRole("button", { name: "Reattribute to another speaker" }));
    const input = screen.getByLabelText("Target speaker id");
    await user.type(input, "s2");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(apiFetch).toHaveBeenCalledWith("/speakers/i1/fragments/5/reattribute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_speaker_id: "s2" }),
    });
  });

  it("segment remove: confirming calls removeSegment's endpoint with reason", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    await user.click(screen.getByRole("button", { name: "Remove segment" }));
    const input = screen.getByLabelText("Reason for removing this segment");
    await user.type(input, "wrong boundary");
    await user.click(screen.getByRole("button", { name: "Confirm remove" }));

    expect(apiFetch).toHaveBeenCalledWith(
      "/segments/i1/seg1?reason=wrong%20boundary",
      { method: "DELETE" },
    );
  });

  it("does not render the segment remove affordance when the line has no segment", () => {
    renderPanel({ ...LINE, segment: null });
    expect(screen.queryByRole("button", { name: "Remove segment" })).not.toBeInTheDocument();
    expect(screen.getByText("Not part of a segment.")).toBeInTheDocument();
  });

  it("lens-item override: correcting a lens item calls overrideLensItem's endpoint/body", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    await user.click(screen.getByRole("button", { name: "Correct" }));
    const input = screen.getByLabelText("Corrected lens item text");
    await user.clear(input);
    await user.type(input, "Ships faster now");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(apiFetch).toHaveBeenCalledWith("/lenses/i1/items/li1/override", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields_overridden: { text: "Ships faster now" } }),
    });
  });

  // --- Manual speaker→person linking (M5.0 Task 6) ---

  it('shows "Identify as person…" for an unlinked speaker, opens the picker, and links an existing person', async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel();

    const openButton = screen.getByRole("button", { name: "Identify as person…" });
    await user.click(openButton);
    expect(screen.getByRole("dialog", { name: "Identify as person" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Jane Doe" }));

    expect(apiFetch).toHaveBeenCalledWith("/resolution/p1/persons/p1/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interview_id: "i1", speaker_id: "s1" }),
    });
  });

  it("does not show the identify affordance for a speakerless line", () => {
    renderPanel({ ...LINE, speaker: null });
    expect(
      screen.queryByRole("button", { name: "Identify as person…" }),
    ).not.toBeInTheDocument();
  });

  it("shows an unlink affordance for an already-linked speaker and calls the unlink endpoint", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue({
      ok: true,
      status: 202,
      json: async () => ({ status: "accepted" }),
    } as Response);

    renderPanel({ ...LINE, person: { person_id: "p1", display_name: "Jane Doe" } });

    expect(
      screen.queryByRole("button", { name: "Identify as person…" }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Unlink person" }));

    expect(apiFetch).toHaveBeenCalledWith("/resolution/p1/persons/p1/unlink", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interview_id: "i1", speaker_id: "s1" }),
    });
  });
});
