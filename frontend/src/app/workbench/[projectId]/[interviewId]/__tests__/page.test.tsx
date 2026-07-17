import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TranscriptPage from "@/app/workbench/[projectId]/[interviewId]/page";
import { useTranscript } from "@/hooks/useTranscript";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { useParams } from "next/navigation";

vi.mock("@/hooks/useTranscript", () => ({
  useTranscript: vi.fn(),
}));

vi.mock("@/hooks/useSentenceHistory", () => ({
  useSentenceHistory: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockParams(projectId: string, interviewId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId, interviewId });
}

const TRANSCRIPT_WITH_SEGMENTS = {
  interview_id: "i1",
  title: "Kickoff call",
  metadata: {},
  lines: [
    {
      fragment_id: "f1",
      sequence_order: 0,
      text: "Let's talk about onboarding.",
      speaker: { speaker_id: "s1", display_name: "Speaker A" },
      person: null,
      utterance_id: "u1",
      segment: { segment_id: "seg1", topic: "Onboarding" },
      entities: [],
      lens_items: [],
      edited: false,
    },
    {
      fragment_id: "f2",
      sequence_order: 1,
      text: "It was confusing at first.",
      speaker: { speaker_id: "s2", display_name: "Speaker B" },
      person: { person_id: "p1", display_name: "Jane Doe" },
      utterance_id: "u1",
      segment: { segment_id: "seg1", topic: "Onboarding" },
      entities: [],
      lens_items: [],
      edited: true,
    },
    {
      fragment_id: "f3",
      sequence_order: 2,
      text: "Now let's discuss pricing.",
      speaker: { speaker_id: "s1", display_name: "Speaker A" },
      person: null,
      utterance_id: "u2",
      segment: { segment_id: "seg2", topic: "Pricing" },
      entities: [],
      lens_items: [],
      edited: false,
    },
  ],
};

describe("TranscriptPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<TranscriptPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("shows the empty state when the interview has no lines", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: { interview_id: "i1", title: "Kickoff call", metadata: {}, lines: [] },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);
    expect(
      screen.getByText("This interview has no transcript lines yet."),
    ).toBeInTheDocument();
  });

  it("renders metadata panel, lines in order, person suffix, edited badge, and segment headings interleaved at the correct positions", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: TRANSCRIPT_WITH_SEGMENTS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);

    // Metadata panel (title + empty-metadata quiet state)
    expect(
      screen.getByRole("heading", { name: "Kickoff call" }),
    ).toBeInTheDocument();
    expect(screen.getByText("No metadata available.")).toBeInTheDocument();

    // Segment headings appear, in document order, before their segment's first line
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    expect(headings).toEqual(["Onboarding", "Pricing"]);

    // Only one "Onboarding" heading — the second line of the same segment
    // does not get a duplicate heading.
    expect(screen.getAllByText("Onboarding")).toHaveLength(1);

    // Person suffix pattern
    expect(screen.getByText("Speaker B (Jane Doe)")).toBeInTheDocument();
    // Edited badge only on the edited line
    expect(screen.getAllByText("edited")).toHaveLength(1);

    // Order: heading "Onboarding" -> line f1 -> line f2 -> heading "Pricing" -> line f3
    const container = screen.getByText("Let's talk about onboarding.").closest("div")!
      .parentElement!.parentElement!;
    const textContent = container.textContent ?? "";
    expect(textContent.indexOf("Onboarding")).toBeLessThan(
      textContent.indexOf("Let's talk about onboarding."),
    );
    expect(textContent.indexOf("It was confusing at first.")).toBeLessThan(
      textContent.indexOf("Pricing"),
    );
    expect(textContent.indexOf("Pricing")).toBeLessThan(
      textContent.indexOf("Now let's discuss pricing."),
    );
  });

  it("indicates utterance grouping: the second line of the same utterance is marked as continuing it", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: TRANSCRIPT_WITH_SEGMENTS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);

    const buttons = screen.getAllByRole("button");
    const first = buttons.find((b) => b.textContent?.includes("Let's talk about onboarding."))!;
    const second = buttons.find((b) => b.textContent?.includes("It was confusing at first."))!;
    const third = buttons.find((b) => b.textContent?.includes("Now let's discuss pricing."))!;

    expect(first).toHaveAttribute("data-continues-utterance", "false");
    expect(second).toHaveAttribute("data-continues-utterance", "true");
    expect(third).toHaveAttribute("data-continues-utterance", "false");
  });

  it("opens the line detail panel when a line is clicked and closes it on close", async () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: TRANSCRIPT_WITH_SEGMENTS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    vi.mocked(useSentenceHistory).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);

    expect(screen.queryByRole("dialog", { name: "Line detail" })).not.toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /Let's talk about onboarding\./ }),
    );

    expect(screen.getByRole("dialog", { name: "Line detail" })).toBeInTheDocument();
    // Lazy history fetch only triggered on open, keyed to this line's sequence_order.
    expect(useSentenceHistory).toHaveBeenCalledWith("i1", 0, true);

    await userEvent.click(screen.getByRole("button", { name: "Close detail panel" }));
    expect(screen.queryByRole("dialog", { name: "Line detail" })).not.toBeInTheDocument();
  });

  it("renders the Workbench / project / interview breadcrumb trail", () => {
    mockParams("p1", "i1");
    vi.mocked(useTranscript).mockReturnValue({
      data: TRANSCRIPT_WITH_SEGMENTS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TranscriptPage />);
    expect(screen.getByRole("link", { name: "Workbench" })).toHaveAttribute(
      "href",
      "/workbench",
    );
    expect(screen.getByRole("link", { name: "p1" })).toHaveAttribute(
      "href",
      "/workbench/p1",
    );
    // Trailing crumb uses the transcript title once loaded.
    expect(screen.getAllByText("Kickoff call").length).toBeGreaterThan(0);
  });
});
