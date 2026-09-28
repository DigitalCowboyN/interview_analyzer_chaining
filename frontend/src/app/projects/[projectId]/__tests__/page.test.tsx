import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import ProjectInterviewsPage from "@/app/projects/[projectId]/page";
import { useInterviews } from "@/hooks/useInterviews";
import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { ApiError } from "@/api/client";

vi.mock("@/hooks/useInterviews", () => ({
  useInterviews: vi.fn(),
}));

const replace = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
  usePathname: vi.fn(),
  useRouter: vi.fn(),
  useSearchParams: vi.fn(),
}));

function mockProjectId(projectId: string, search = "") {
  vi.mocked(useParams).mockReturnValue({ projectId });
  vi.mocked(usePathname).mockReturnValue(`/projects/${projectId}`);
  vi.mocked(useRouter).mockReturnValue({ replace } as never);
  vi.mocked(useSearchParams).mockReturnValue(new URLSearchParams(search) as never);
}

// useLiveInvalidation (Task 5) reaches for the real useQueryClient, so page
// renders need a provider — mirrors the sibling transcript page test.
function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return render(<ProjectInterviewsPage />, { wrapper: Wrapper });
}

describe("ProjectInterviewsPage (interviews)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    replace.mockClear();
  });

  it("shows the loading state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the empty state when the project has no interviews", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByText("No interviews yet.")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    renderPage();
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it("renders interviews and links each to its transcript route (Task 4's route)", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [
        {
          interview_id: "i1",
          title: "Kickoff call",
          created_at: "2026-01-01T00:00:00Z",
          fragment_count: 42,
          participants: [],
          insight_counts: {},
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByRole("link", { name: /Kickoff call/ }),
    ).toHaveAttribute("href", "/projects/p1/interviews/i1");
  });

  const MIXED_INTERVIEWS = [
    {
      interview_id: "i1",
      title: "Kickoff call",
      created_at: "2026-01-01T00:00:00Z",
      fragment_count: 42,
      participants: [],
      insight_counts: {},
    },
    {
      interview_id: "i2",
      title: "Follow-up call",
      created_at: "2026-01-02T00:00:00Z",
      fragment_count: 10,
      participants: [],
      insight_counts: {},
    },
    {
      interview_id: "i3",
      title: "Empty upload",
      created_at: "2026-01-03T00:00:00Z",
      fragment_count: 0,
      participants: [],
      insight_counts: {},
    },
  ];

  it("hides empty interviews by default; clicking the toggle replaces the URL with ?empty=1", async () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: MIXED_INTERVIEWS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getAllByRole("link")).toHaveLength(2);
    const toggle = screen.getByRole("button", { name: "Show 1 empty" });

    const user = userEvent.setup();
    await user.click(toggle);

    expect(replace).toHaveBeenCalledWith("/projects/p1?empty=1", { scroll: false });
  });

  it("preserves an existing search param when the toggle sets ?empty=1", async () => {
    mockProjectId("p1", "sort=recent");
    vi.mocked(useInterviews).mockReturnValue({
      data: MIXED_INTERVIEWS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    const toggle = screen.getByRole("button", { name: "Show 1 empty" });
    await userEvent.click(toggle);

    expect(replace).toHaveBeenCalledWith("/projects/p1?sort=recent&empty=1", { scroll: false });
  });

  it("reads the toggle state from ?empty=1: shows empty rows and 'Hide empty interviews'", () => {
    mockProjectId("p1", "empty=1");
    vi.mocked(useInterviews).mockReturnValue({
      data: MIXED_INTERVIEWS,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getAllByRole("link")).toHaveLength(3);
    expect(screen.getByRole("button", { name: "Hide empty interviews" })).toBeInTheDocument();
  });

  it("shows an explanatory message instead of an empty list when every interview has no lines", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [
        {
          interview_id: "i1",
          title: "Empty upload",
          created_at: "2026-01-01T00:00:00Z",
          fragment_count: 0,
          participants: [],
          insight_counts: {},
        },
        {
          interview_id: "i2",
          title: "Another empty upload",
          created_at: "2026-01-02T00:00:00Z",
          fragment_count: 0,
          participants: [],
          insight_counts: {},
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(
      screen.getByText("All 2 interviews in this project have no transcript lines yet."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show 2 empty" })).toBeInTheDocument();
  });

  it("shows the singular all-empty message for a single empty interview", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [
        {
          interview_id: "i1",
          title: "Empty upload",
          created_at: "2026-01-01T00:00:00Z",
          fragment_count: 0,
          participants: [],
          insight_counts: {},
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByText("All 1 interview in this project has no transcript lines yet."),
    ).toBeInTheDocument();
  });

  it("shows a not-found message with a link back to all projects on a 404", () => {
    mockProjectId("missing-project");
    vi.mocked(useInterviews).mockReturnValue({
      data: undefined,
      isError: true,
      error: new ApiError(404, "Project not found"),
      isLoading: false,
    } as never);

    renderPage();
    expect(screen.getByText(/Project not found\./)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "All projects" })).toHaveAttribute(
      "href",
      "/",
    );
  });

  it("calls useInterviews with the project id from the route params", () => {
    mockProjectId("my project");
    vi.mocked(useInterviews).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(useInterviews).toHaveBeenCalledWith("my project");
  });
});
