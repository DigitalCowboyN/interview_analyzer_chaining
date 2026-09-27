import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import TranscriptPage from "@/app/projects/[projectId]/interviews/[interviewId]/page";
import { useTranscript } from "@/hooks/useTranscript";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { useInsights } from "@/hooks/useInsights";
import { useInterviews } from "@/hooks/useInterviews";
import { useProject } from "@/hooks/useProjects";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import type { Insight } from "@/lib/insights";

vi.mock("@/hooks/useTranscript", () => ({
  useTranscript: vi.fn(),
}));

vi.mock("@/hooks/useSentenceHistory", () => ({
  useSentenceHistory: vi.fn(),
}));

vi.mock("@/hooks/useInsights", () => ({
  useInsights: vi.fn(),
}));

vi.mock("@/hooks/useInterviews", () => ({
  useInterviews: vi.fn(),
}));

vi.mock("@/hooks/useProjects", () => ({
  useProject: vi.fn(),
}));

const replace = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
  usePathname: vi.fn(),
  useRouter: vi.fn(),
  useSearchParams: vi.fn(),
}));

function mockNav(search: string) {
  vi.mocked(useParams).mockReturnValue({ projectId: "p1", interviewId: "i1" });
  vi.mocked(usePathname).mockReturnValue("/projects/p1/interviews/i1");
  vi.mocked(useRouter).mockReturnValue({ replace } as never);
  vi.mocked(useSearchParams).mockReturnValue(new URLSearchParams(search) as never);
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return render(<TranscriptPage />, { wrapper: Wrapper });
}

const TRANSCRIPT = {
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
      utterance_id: null,
      segment: null,
      entities: [],
      lens_items: [],
      edited: false,
    },
    {
      fragment_id: "f2",
      sequence_order: 1,
      text: "It was confusing at first.",
      speaker: { speaker_id: "s2", display_name: "Speaker B" },
      person: null,
      utterance_id: null,
      segment: null,
      entities: [],
      lens_items: [],
      edited: false,
    },
  ],
};

const INSIGHTS: Insight[] = [
  {
    item_id: "d1",
    node_type: "Decision",
    lens: "meeting_minutes",
    text: "Ship CSV export",
    confidence: 0.92,
    locked: true,
    supporting_fragment_ids: ["f1"],
  },
];

describe("TranscriptPage", () => {
  beforeEach(() => {
    vi.mocked(useTranscript).mockReturnValue({
      data: TRANSCRIPT,
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
    vi.mocked(useInsights).mockReturnValue({
      data: INSIGHTS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    vi.mocked(useInterviews).mockReturnValue({
      data: [{ interview_id: "i1", title: "Kickoff call", created_at: "2026-09-27T00:00:00Z" }],
    } as never);
    vi.mocked(useProject).mockReturnValue({
      project: { project_id: "p1", interview_count: 1, kind: "real", suite: null },
      isLoading: false,
    } as never);
    replace.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the Insights panel with a group heading when no line param", () => {
    mockNav("");
    renderPage();
    expect(screen.getByRole("heading", { name: "Insights" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Decisions (1)" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Line detail" })).not.toBeInTheDocument();
  });

  it("?line=f1 shows the Line detail dialog instead of Insights", () => {
    mockNav("line=f1");
    renderPage();
    expect(screen.getByRole("dialog", { name: "Line detail" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Decisions (1)" })).not.toBeInTheDocument();
  });

  it("?insight=d1 highlights its supporting fragment lines only", () => {
    mockNav("insight=d1");
    renderPage();
    const f1 = document.getElementById("line-f1")!;
    const f2 = document.getElementById("line-f2")!;
    expect(f1).toHaveAttribute("data-highlighted", "true");
    expect(f2).not.toHaveAttribute("data-highlighted", "true");
  });

  it("ignores unknown line/insight ids without throwing, showing Insights", () => {
    mockNav("line=gone&insight=gone");
    expect(() => renderPage()).not.toThrow();
    expect(screen.getByRole("heading", { name: "Insights" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Line detail" })).not.toBeInTheDocument();
    const f1 = document.getElementById("line-f1")!;
    const f2 = document.getElementById("line-f2")!;
    expect(f1).not.toHaveAttribute("data-highlighted", "true");
    expect(f2).not.toHaveAttribute("data-highlighted", "true");
  });

  it("clicking a transcript line replaces the URL with ?line=<fragment_id>", async () => {
    mockNav("");
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: /Let's talk about onboarding\./ }));
    expect(replace).toHaveBeenCalledWith("/projects/p1/interviews/i1?line=f1", { scroll: false });
  });

  it("closing the Line detail panel removes ?line and shows Insights again", async () => {
    mockNav("line=f1&insight=d1");
    renderPage();
    await userEvent.click(screen.getByRole("button", { name: "Close detail panel" }));
    expect(replace).toHaveBeenCalledWith("/projects/p1/interviews/i1?insight=d1", { scroll: false });
  });

  it("shows an insights error message while the transcript still renders", () => {
    mockNav("");
    vi.mocked(useInsights).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);
    renderPage();
    expect(screen.getByText("Couldn’t load insights.")).toBeInTheDocument();
    expect(screen.getByText("Let's talk about onboarding.")).toBeInTheDocument();
    expect(screen.getByText("It was confusing at first.")).toBeInTheDocument();
  });
});
