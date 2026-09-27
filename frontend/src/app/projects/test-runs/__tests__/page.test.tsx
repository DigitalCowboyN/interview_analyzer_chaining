import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TestRunsPage from "@/app/projects/test-runs/page";
import { useTestRunInterviews } from "@/hooks/useTestRunInterviews";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

vi.mock("@/hooks/useTestRunInterviews", () => ({
  useTestRunInterviews: vi.fn(),
}));

const replace = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(),
  useRouter: vi.fn(),
  useSearchParams: vi.fn(),
}));

function mockNav(search = "") {
  vi.mocked(usePathname).mockReturnValue("/projects/test-runs");
  vi.mocked(useRouter).mockReturnValue({ replace } as never);
  vi.mocked(useSearchParams).mockReturnValue(new URLSearchParams(search) as never);
}

const ROWS = [
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
];

const ROWS_WITH_EMPTY = [
  ...ROWS,
  {
    interview_id: "d",
    project_id: "smoke-3",
    suite: "smoke",
    title: "Smoke run 3 (empty)",
    created_at: "2026-09-27T13:00:00Z",
    fragment_count: 0,
    participants: [],
    insight_counts: {},
  },
];

describe("TestRunsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    replace.mockClear();
  });

  it("groups test-run interviews by suite, one <details> per suite", () => {
    mockNav();
    vi.mocked(useTestRunInterviews).mockReturnValue({
      data: ROWS,
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

  it("hides zero-line interviews by default, behind a 'Show N empty' toggle", () => {
    mockNav();
    vi.mocked(useTestRunInterviews).mockReturnValue({
      data: ROWS_WITH_EMPTY,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TestRunsPage />);

    expect(screen.queryByText(/Smoke run 3/)).not.toBeInTheDocument();
    expect(screen.getByText("smoke (2)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show 1 empty" })).toBeInTheDocument();
  });

  it("clicking the toggle replaces the URL with ?empty=1", async () => {
    mockNav();
    vi.mocked(useTestRunInterviews).mockReturnValue({
      data: ROWS_WITH_EMPTY,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TestRunsPage />);
    await userEvent.click(screen.getByRole("button", { name: "Show 1 empty" }));

    expect(replace).toHaveBeenCalledWith("/projects/test-runs?empty=1", { scroll: false });
  });

  it("?empty=1 shows the zero-line interviews and 'Hide empty interviews'", () => {
    mockNav("empty=1");
    vi.mocked(useTestRunInterviews).mockReturnValue({
      data: ROWS_WITH_EMPTY,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    render(<TestRunsPage />);

    expect(screen.getByText(/Smoke run 3/)).toBeInTheDocument();
    expect(screen.getByText("smoke (3)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide empty interviews" })).toBeInTheDocument();
  });
});
