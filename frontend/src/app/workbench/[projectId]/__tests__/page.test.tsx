import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import ProjectInterviewsPage from "@/app/workbench/[projectId]/page";
import { useInterviews } from "@/hooks/useInterviews";
import { useParams } from "next/navigation";

vi.mock("@/hooks/useInterviews", () => ({
  useInterviews: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: vi.fn(),
}));

function mockProjectId(projectId: string) {
  vi.mocked(useParams).mockReturnValue({ projectId });
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
        },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByRole("link", { name: /Kickoff call/ }),
    ).toHaveAttribute("href", "/workbench/p1/i1");
  });

  it("renders the Workbench / project breadcrumb trail", () => {
    mockProjectId("p1");
    vi.mocked(useInterviews).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByRole("link", { name: "Workbench" })).toHaveAttribute(
      "href",
      "/workbench",
    );
    expect(screen.getByText("p1")).toBeInTheDocument();
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
