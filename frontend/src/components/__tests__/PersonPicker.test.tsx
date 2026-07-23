import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { PersonPicker } from "@/components/PersonPicker";
import { usePersons, usePersonId } from "@/hooks/usePersons";
import { queryKeys } from "@/hooks/queryKeys";
import { apiFetch, ApiError } from "@/api/client";

vi.mock("@/hooks/usePersons", () => ({
  usePersons: vi.fn(),
  usePersonId: vi.fn(),
}));

vi.mock("@/api/client", async () => {
  const actual = await vi.importActual<typeof import("@/api/client")>("@/api/client");
  return { ...actual, apiFetch: vi.fn() };
});

// Small injected poll interval (mirrors flows.test.tsx's POLL_OPTIONS) so
// tests awaiting settle don't pay the production 2s x 10 poll cost.
const POLL_OPTIONS = { pollIntervalMs: 10, maxPollAttempts: 10 };
const TRANSCRIPT_QUERY_KEY = queryKeys.transcript("i1");

/**
 * Renders the picker with a transcript query the intent wrapper can poll:
 * the queryFn reads `linkedPersonId.current` each refetch, so the test can
 * flip it once the link request has fired (simulating the backend having
 * caught up) to let the poll settle deterministically.
 */
function renderPicker(onLinked = vi.fn(), onClose = vi.fn()) {
  const linkedPersonId = { current: null as string | null };
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryDefaults(TRANSCRIPT_QUERY_KEY, {
    queryFn: async () => ({
      lines: [
        {
          fragment_id: "f1",
          sequence_order: 0,
          speaker: { speaker_id: "s1", display_name: "Speaker A" },
          person: linkedPersonId.current
            ? { person_id: linkedPersonId.current, display_name: "Linked" }
            : null,
          segment: null,
          lens_items: [],
          edited: false,
        },
      ],
    }),
  });
  client.setQueryData(TRANSCRIPT_QUERY_KEY, { lines: [] });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  const utils = render(
    <PersonPicker
      projectId="proj1"
      interviewId="i1"
      speakerId="s1"
      onClose={onClose}
      onLinked={onLinked}
      pollOptions={POLL_OPTIONS}
    />,
    { wrapper: Wrapper },
  );
  return { ...utils, linkedPersonId };
}

function mockAccepted() {
  return {
    ok: true,
    status: 202,
    json: async () => ({ status: "accepted" }),
  } as Response;
}

describe("PersonPicker", () => {
  beforeEach(() => {
    vi.mocked(apiFetch).mockReset();
    vi.mocked(usePersons).mockReturnValue({
      data: [
        { person_id: "p1", display_name: "Jane Doe", speaker_count: 1, interview_count: 1 },
        { person_id: "p2", display_name: "John Smith", speaker_count: 1, interview_count: 1 },
      ],
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    vi.mocked(usePersonId).mockReturnValue({
      derivePersonId: vi.fn().mockResolvedValue("person-new"),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("lists the project's persons", () => {
    renderPicker();
    expect(screen.getByRole("button", { name: "Jane Doe" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "John Smith" })).toBeInTheDocument();
  });

  it("shows a loading state while persons are loading", () => {
    vi.mocked(usePersons).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as never);
    renderPicker();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows an empty state when the project has no persons yet", () => {
    vi.mocked(usePersons).mockReturnValue({
      data: [],
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    renderPicker();
    expect(screen.getByText("No persons yet.")).toBeInTheDocument();
  });

  it("linking to an existing person calls link WITHOUT display_name", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue(mockAccepted());
    const onLinked = vi.fn();
    const { linkedPersonId } = renderPicker(onLinked);

    await user.click(screen.getByRole("button", { name: "Jane Doe" }));

    expect(apiFetch).toHaveBeenCalledWith("/resolution/proj1/persons/p1/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ interview_id: "i1", speaker_id: "s1" }),
    });

    // Simulate the backend having caught up so the bounded poll settles.
    linkedPersonId.current = "p1";
    await waitFor(() => expect(onLinked).toHaveBeenCalled());
  });

  it("create-new flow calls derivation THEN link with display_name", async () => {
    const user = userEvent.setup();
    vi.mocked(apiFetch).mockResolvedValue(mockAccepted());
    const derivePersonId = vi.fn().mockResolvedValue("person-new");
    vi.mocked(usePersonId).mockReturnValue({ derivePersonId });
    const onLinked = vi.fn();
    const { linkedPersonId } = renderPicker(onLinked);

    await user.type(screen.getByLabelText("New person display name"), "New Person");
    await user.click(screen.getByRole("button", { name: "Create and link" }));

    await waitFor(() => expect(derivePersonId).toHaveBeenCalledWith("proj1", "New Person"));
    expect(apiFetch).toHaveBeenCalledWith("/resolution/proj1/persons/person-new/link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        interview_id: "i1",
        speaker_id: "s1",
        display_name: "New Person",
      }),
    });

    linkedPersonId.current = "person-new";
    await waitFor(() => expect(onLinked).toHaveBeenCalled());

    // Derivation must be called before the link request — the frontend never
    // derives the id itself.
    const deriveOrder = derivePersonId.mock.invocationCallOrder[0];
    const linkOrder = vi.mocked(apiFetch).mock.invocationCallOrder[0];
    expect(deriveOrder).toBeLessThan(linkOrder);
  });

  it("surfaces a notice (not an unhandled rejection) when derivePersonId rejects", async () => {
    const user = userEvent.setup();
    const unhandledRejections: unknown[] = [];
    const onUnhandledRejection = (event: PromiseRejectionEvent) => {
      unhandledRejections.push(event.reason);
    };
    window.addEventListener("unhandledrejection", onUnhandledRejection);

    const derivePersonId = vi.fn().mockRejectedValue(new ApiError(404, "Project not found"));
    vi.mocked(usePersonId).mockReturnValue({ derivePersonId });
    const onLinked = vi.fn();
    renderPicker(onLinked);

    await user.type(screen.getByLabelText("New person display name"), "New Person");
    await user.click(screen.getByRole("button", { name: "Create and link" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Project not found");
    expect(onLinked).not.toHaveBeenCalled();
    expect(apiFetch).not.toHaveBeenCalled();

    // Give any unhandled-rejection microtask a turn to fire before asserting.
    await Promise.resolve();
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    expect(unhandledRejections).toHaveLength(0);
  });

  it("does not submit create-new with a blank name", async () => {
    renderPicker();
    expect(screen.getByRole("button", { name: "Create and link" })).toBeDisabled();
  });

  it("calls onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderPicker(vi.fn(), onClose);
    await user.click(screen.getByRole("button", { name: "Close person picker" }));
    expect(onClose).toHaveBeenCalled();
  });
});
