import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import ProjectPersonsPage from "@/app/gallery/persons/[projectId]/page";
import { usePersons } from "@/hooks/usePersons";
import { useParams } from "next/navigation";

vi.mock("@/hooks/usePersons", () => ({
  usePersons: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockProjectId(projectId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId });
}

describe("ProjectPersonsPage (person cards)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(usePersons).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonsPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the empty state when the project has no persons", () => {
    mockProjectId("p1");
    vi.mocked(usePersons).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonsPage />);
    expect(screen.getByText("No persons yet.")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(usePersons).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<ProjectPersonsPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("renders person cards and links each to its core view route", () => {
    mockProjectId("p1");
    vi.mocked(usePersons).mockReturnValue({
      data: [
        { person_id: "person1", display_name: "Jane Doe", speaker_count: 2, interview_count: 1 },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<ProjectPersonsPage />);
    expect(screen.getByRole("link", { name: /Jane Doe/ })).toHaveAttribute(
      "href",
      "/gallery/persons/p1/person1",
    );
  });
});
