import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { LineDetailPanel } from "@/components/LineDetailPanel";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import type { TranscriptLineData } from "@/hooks/useTranscript";

vi.mock("@/hooks/useSentenceHistory", () => ({
  useSentenceHistory: vi.fn(),
}));

const LINE: TranscriptLineData = {
  fragment_id: "f1",
  sequence_order: 5,
  text: "Full sentence text",
  speaker: { speaker_id: "s1", display_name: "Speaker A" },
  person: null,
  utterance_id: null,
  segment: null,
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

describe("LineDetailPanel", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders full text, entities, and lens items (lens, type, text, confidence, lock state)", () => {
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(
      <LineDetailPanel interviewId="i1" line={LINE} onClose={vi.fn()} />,
    );

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
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<LineDetailPanel interviewId="i1" line={LINE} onClose={vi.fn()} />);

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

    render(<LineDetailPanel interviewId="i1" line={LINE} onClose={vi.fn()} />);

    expect(screen.getByText("SentenceEdited")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);
    const onClose = vi.fn();

    render(<LineDetailPanel interviewId="i1" line={LINE} onClose={onClose} />);
    screen.getByRole("button", { name: "Close detail panel" }).click();
    expect(onClose).toHaveBeenCalled();
  });
});
