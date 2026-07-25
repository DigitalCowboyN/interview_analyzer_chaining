import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import PersonDetailPage from "@/app/gallery/persons/[projectId]/[personId]/page";
import { usePersonDetail } from "@/hooks/usePersonDetail";
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { useParams } from "next/navigation";

vi.mock("@/hooks/usePersonDetail", () => ({
  usePersonDetail: vi.fn(),
}));

// Mocked so the page test doesn't open a real EventSource / need a
// QueryClient provider; the real LiveIndicator still renders off its status.
vi.mock("@/hooks/useLiveInvalidation", () => ({
  useLiveInvalidation: vi.fn(() => "idle"),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockParams(projectId: string, personId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId, personId });
}

describe("PersonDetailPage (person core view)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonDetail).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<PersonDetailPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonDetail).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<PersonDetailPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("renders identity facts and a loose link to the persona profile (navigation, not an embed)", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonDetail).mockReturnValue({
      data: {
        person_id: "person1",
        display_name: "Jane Doe",
        links: [
          {
            interview_id: "i1",
            interview_title: "Kickoff call",
            speaker_id: "s1",
            speaker_display_name: "Speaker A",
          },
        ],
        contributes_to_persona: true,
      },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<PersonDetailPage />);
    expect(screen.getByRole("heading", { name: "Jane Doe" })).toBeInTheDocument();
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
    // Separate-route-tree assertion: the persona link navigates to /gallery/personas/..., not /gallery/persons/...
    expect(screen.getByRole("link", { name: /View persona profile/ })).toHaveAttribute(
      "href",
      "/gallery/personas/p1/person1",
    );
  });

  it("calls usePersonDetail with the project/person ids from route params", () => {
    mockParams("p1", "person1");
    vi.mocked(usePersonDetail).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<PersonDetailPage />);
    expect(usePersonDetail).toHaveBeenCalledWith("p1", "person1");
  });

  it("wires the live-updates indicator with the project/person scope", () => {
    mockParams("p1", "person1");
    vi.mocked(useLiveInvalidation).mockReturnValue("live");
    vi.mocked(usePersonDetail).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<PersonDetailPage />);
    expect(screen.getByText(/Live updates/i)).toBeInTheDocument();
    expect(useLiveInvalidation).toHaveBeenCalledWith({ projectId: "p1", personId: "person1" });
  });
});
