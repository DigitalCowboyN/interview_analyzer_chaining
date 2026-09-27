import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import TestRunsPage from "@/app/projects/test-runs/page";
import { useTestRunInterviews } from "@/hooks/useTestRunInterviews";

vi.mock("@/hooks/useTestRunInterviews", () => ({
  useTestRunInterviews: vi.fn(),
}));

describe("TestRunsPage", () => {
  it("groups test-run interviews by suite, one <details> per suite", () => {
    vi.mocked(useTestRunInterviews).mockReturnValue({
      data: [
        {
          interview_id: "a",
          project_id: "smoke-1",
          suite: "smoke",
          title: "Smoke run 1",
          created_at: "2026-09-27T10:00:00Z",
          fragment_count: 5,
          participants: [],
          insight_counts: {},
        },
        {
          interview_id: "b",
          project_id: "smoke-2",
          suite: "smoke",
          title: "Smoke run 2",
          created_at: "2026-09-27T11:00:00Z",
          fragment_count: 3,
          participants: [],
          insight_counts: {},
        },
        {
          interview_id: "c",
          project_id: "ui-smoke-1",
          suite: "ui-smoke",
          title: "UI smoke run 1",
          created_at: "2026-09-27T12:00:00Z",
          fragment_count: 7,
          participants: [],
          insight_counts: {},
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TestRunsPage />);

    expect(screen.getAllByRole("group")).toHaveLength(2);
    expect(screen.getByText("smoke (2)")).toBeInTheDocument();
    expect(screen.getByText("ui-smoke (1)")).toBeInTheDocument();

    expect(screen.getByRole("link", { name: /Smoke run 1/ })).toHaveAttribute(
      "href",
      "/projects/smoke-1/interviews/a",
    );
  });
});
