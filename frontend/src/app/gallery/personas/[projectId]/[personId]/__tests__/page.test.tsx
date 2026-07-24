import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import PersonaDetailPage from "@/app/gallery/personas/[projectId]/[personId]/page";
import { usePersonaDetail } from "@/hooks/usePersonaDetail";
import { useParams } from "next/navigation";

vi.mock("@/hooks/usePersonaDetail", () => ({
  usePersonaDetail: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockParams(projectId: string, personId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId, personId });
}

describe("PersonaDetailPage (persona core view)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonaDetail).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<PersonaDetailPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonaDetail).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<PersonaDetailPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("renders the persona core view with dimension-grouped items and provenance", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonaDetail).mockReturnValue({
      data: {
        person_id: "person1",
        display_name: "Jane Doe",
        dimensions: {
          traits: [
            {
              item_id: "t1",
              text: "Detail-oriented",
              confidence: 0.9,
              interview_id: "i1",
              interview_title: "Kickoff call",
            },
          ],
          goals: [],
          pain_points: [],
          notable_quotes: [],
        },
      },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<PersonaDetailPage />);
    expect(screen.getByRole("heading", { name: "Jane Doe" })).toBeInTheDocument();
    expect(screen.getByText("Detail-oriented")).toBeInTheDocument();
    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
  });

  it("calls usePersonaDetail with the project/person ids from route params", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonaDetail).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<PersonaDetailPage />);
    expect(usePersonaDetail).toHaveBeenCalledWith("p1", "person1");
  });
});
