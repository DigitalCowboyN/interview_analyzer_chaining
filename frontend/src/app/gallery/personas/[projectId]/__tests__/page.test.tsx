import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ProjectPersonasPage from "@/app/gallery/personas/[projectId]/page";
import { usePersonaCards } from "@/hooks/usePersonaCards";
import { useInterviews } from "@/hooks/useInterviews";
import { useParams } from "next/navigation";

vi.mock("@/hooks/usePersonaCards", () => ({
  usePersonaCards: vi.fn(),
}));

vi.mock("@/hooks/useInterviews", () => ({
  useInterviews: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockProjectId(projectId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId });
}

describe("ProjectPersonasPage (persona cards)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({ data: [] } as never);
    vi.mocked(usePersonaCards).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonasPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({ data: [] } as never);
    vi.mocked(usePersonaCards).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<ProjectPersonasPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("shows a helpful empty state naming the CLI when the project has no persona-lens run yet", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({ data: [] } as never);
    vi.mocked(usePersonaCards).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonasPage />);
    expect(
      screen.getByText(/No persona profiles yet\. Run the persona lens for an interview:/),
    ).toBeInTheDocument();
    expect(screen.getByText("python -m src.lens <interview_id> persona")).toBeInTheDocument();
  });

  it("renders persona cards and the per-interview filter once data has loaded", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [{ interview_id: "i1", title: "Kickoff call", created_at: "2026-01-01", fragment_count: 10 }],
    } as never);
    vi.mocked(usePersonaCards).mockReturnValue({
      data: [
        {
          person_id: "p1",
          display_name: "Jane Doe",
          trait_count: 1,
          goal_count: 1,
          pain_point_count: 0,
          quote_count: 0,
          representative_quote: null,
          interview_ids: ["i1"],
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonasPage />);
    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByLabelText("Filter by interview")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Jane Doe/ })).toHaveAttribute(
      "href",
      "/gallery/personas/p1/p1",
    );
  });

  it("shows the v1-honesty note about seeded persona profiles", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({ data: [] } as never);
    vi.mocked(usePersonaCards).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonasPage />);
    expect(
      screen.getByText("Persona profiles are currently seeded from per-person contributions."),
    ).toBeInTheDocument();
  });
});
