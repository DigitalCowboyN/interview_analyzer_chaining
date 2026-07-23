import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import WorklistPage from "@/app/gallery/worklist/page";
import { useWorklist } from "@/hooks/useWorklist";
import { useSearchParams } from "next/navigation";

vi.mock("@/hooks/useWorklist", () => ({
  useWorklist: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: vi.fn(),
}));

function mockProjectParam(project: string | null) {
  vi.mocked(useSearchParams).mockReturnValue(new URLSearchParams(
    project ? { project } : {},
  ) as never);
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  return render(<WorklistPage />, { wrapper: Wrapper });
}

const EMPTY_DATA = {
  lens_items: [],
  claims: [],
  entity_merge_suggestions: [],
  person_link_suggestions: [],
  flags: [],
};

describe("WorklistPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a prompt state when the project query param is missing", () => {
    mockProjectParam(null);
    vi.mocked(useWorklist).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByText("Select a project from the Gallery to view its worklist."),
    ).toBeInTheDocument();
    expect(useWorklist).toHaveBeenCalledWith("");
  });

  it("shows a prompt state when the project query param is empty", () => {
    mockProjectParam("");
    vi.mocked(useWorklist).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByText("Select a project from the Gallery to view its worklist."),
    ).toBeInTheDocument();
  });

  it("shows the loading state via StateGate", () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the error state via StateGate", () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as never);

    renderPage();
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
  });

  it('shows "Nothing to review" when all four row arrays are empty and there are no flags', () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: EMPTY_DATA,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByText("Nothing to review.")).toBeInTheDocument();
  });

  it("renders worklist rows when the project has review items", () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: {
        ...EMPTY_DATA,
        claims: [
          {
            interview_id: "i1",
            claim_id: "c1",
            text: "We ship weekly.",
            kind: "assertion",
            confidence: 0.5,
            reason: "low_confidence",
          },
        ],
      },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByText("We ship weekly.")).toBeInTheDocument();
    expect(screen.queryByText("Nothing to review.")).not.toBeInTheDocument();
  });

  it("does not treat a degraded-but-otherwise-empty worklist as the empty state (banner still renders)", () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: { ...EMPTY_DATA, flags: ["embedding_unavailable"] },
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(
      screen.getByText("suggestions degraded — embedding provider unavailable"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Nothing to review.")).not.toBeInTheDocument();
  });

  it("renders the Gallery / project / Worklist breadcrumb trail", () => {
    mockProjectParam("proj1");
    vi.mocked(useWorklist).mockReturnValue({
      data: EMPTY_DATA,
      isLoading: false,
      isError: false,
      error: null,
    } as never);

    renderPage();
    expect(screen.getByRole("link", { name: "Gallery" })).toHaveAttribute("href", "/gallery");
    expect(screen.getByText("proj1")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Worklist" })).toBeInTheDocument();
  });
});
