import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import WorkbenchPage from "@/app/workbench/page";
import { useProjects } from "@/hooks/useProjects";

vi.mock("@/hooks/useProjects", () => ({
  useProjects: vi.fn(),
}));

describe("WorkbenchPage (projects)", () => {
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

    render(<WorkbenchPage />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the empty state when there are no projects", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<WorkbenchPage />);
    expect(screen.getByText("No projects yet.")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    render(<WorkbenchPage />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("renders projects and links each to its interviews screen", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [{ project_id: "p1", interview_count: 2 }],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<WorkbenchPage />);
    expect(screen.getByRole("link", { name: /p1/ })).toHaveAttribute(
      "href",
      "/workbench/p1",
    );
  });

  it("renders the Workbench breadcrumb", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<WorkbenchPage />);
    expect(screen.getByText("Workbench")).toBeInTheDocument();
  });
});
