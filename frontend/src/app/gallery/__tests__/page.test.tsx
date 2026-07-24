import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import GalleryPage from "@/app/gallery/page";
import { useProjects } from "@/hooks/useProjects";

vi.mock("@/hooks/useProjects", () => ({
  useProjects: vi.fn(),
}));

describe("GalleryPage (home)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the loading state via StateGate", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    render(<GalleryPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the empty state when there are no projects", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<GalleryPage />);
    expect(screen.getByText("No projects yet.")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<GalleryPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("does not show the Personas/Persons/Worklist areas until a project is selected", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [{ project_id: "p1", interview_count: 2 }],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<GalleryPage />);
    expect(screen.queryByText("Personas")).not.toBeInTheDocument();
    expect(screen.queryByText("Persons")).not.toBeInTheDocument();
    expect(screen.queryByText("Worklist")).not.toBeInTheDocument();
  });

  it("shows Personas / Persons / Worklist links to the selected project, each with the correct navigation target", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [{ project_id: "p1", interview_count: 2 }],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<GalleryPage />);
    fireEvent.change(screen.getByLabelText("Project"), { target: { value: "p1" } });

    expect(screen.getByRole("link", { name: /Personas/ })).toHaveAttribute(
      "href",
      "/gallery/personas/p1",
    );
    expect(screen.getByRole("link", { name: /Persons/ })).toHaveAttribute(
      "href",
      "/gallery/persons/p1",
    );
    expect(screen.getByRole("link", { name: /Worklist/ })).toHaveAttribute(
      "href",
      "/gallery/worklist?project=p1",
    );
  });
});
