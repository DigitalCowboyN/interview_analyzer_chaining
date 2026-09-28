import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import HomePage from "@/app/page";
import { useProjects } from "@/hooks/useProjects";

vi.mock("@/hooks/useProjects", () => ({
  useProjects: vi.fn(),
}));

describe("HomePage (landing)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows real projects as cards and bundles test runs into one card", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [
        { project_id: "samples", interview_count: 4, kind: "real", suite: null },
        { project_id: "smoke-1", interview_count: 2, kind: "test", suite: "smoke" },
        { project_id: "ui-smoke-1", interview_count: 1, kind: "test", suite: "ui-smoke" },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<HomePage />);

    const samplesLink = screen.getByRole("link", { name: /Samples/ });
    expect(samplesLink).toHaveAttribute("href", "/projects/samples");
    expect(samplesLink).toHaveTextContent("4 interviews");

    const testRunsLink = screen.getByRole("link", { name: /Test runs/ });
    expect(testRunsLink).toHaveAttribute("href", "/projects/test-runs");
    expect(testRunsLink).toHaveTextContent("3 interviews");

    const links = screen.getAllByRole("link");
    for (const link of links) {
      expect(link.textContent ?? "").not.toContain("smoke-1");
    }
  });

  it("shows no Test runs card when there are no test projects", () => {
    vi.mocked(useProjects).mockReturnValue({
      data: [{ project_id: "samples", interview_count: 4, kind: "real", suite: null }],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<HomePage />);

    expect(screen.queryByRole("link", { name: /Test runs/ })).not.toBeInTheDocument();
  });
});
