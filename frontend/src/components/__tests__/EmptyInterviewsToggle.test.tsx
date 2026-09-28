import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmptyInterviewsToggle, splitEmpty } from "@/components/EmptyInterviewsToggle";

describe("splitEmpty", () => {
  it("splits items by fragment_count, counting the empties", () => {
    const items = [
      { fragment_count: 3 },
      { fragment_count: 0 },
      { fragment_count: 5 },
    ];
    expect(splitEmpty(items)).toEqual({
      withLines: [{ fragment_count: 3 }, { fragment_count: 5 }],
      emptyCount: 1,
    });
  });
});

describe("EmptyInterviewsToggle", () => {
  it("renders nothing when emptyCount is 0", () => {
    const { container } = render(
      <EmptyInterviewsToggle emptyCount={0} allEmpty={false} showEmpty={false} onToggle={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows 'Show N empty' when hidden, and calls onToggle when clicked", async () => {
    const onToggle = vi.fn();
    render(<EmptyInterviewsToggle emptyCount={3} allEmpty={false} showEmpty={false} onToggle={onToggle} />);
    const button = screen.getByRole("button", { name: "Show 3 empty" });
    await userEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("shows 'Hide empty interviews' when showing", () => {
    render(<EmptyInterviewsToggle emptyCount={3} allEmpty={false} showEmpty onToggle={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Hide empty interviews" })).toBeInTheDocument();
  });

  it("shows the plural all-empty message above the toggle, scoped to 'this project' by default", () => {
    render(<EmptyInterviewsToggle emptyCount={2} allEmpty showEmpty={false} onToggle={vi.fn()} />);
    expect(
      screen.getByText("All 2 interviews in this project have no transcript lines yet."),
    ).toBeInTheDocument();
  });

  it("shows the singular all-empty message with correct verb agreement", () => {
    render(<EmptyInterviewsToggle emptyCount={1} allEmpty showEmpty={false} onToggle={vi.fn()} />);
    expect(
      screen.getByText("All 1 interview in this project has no transcript lines yet."),
    ).toBeInTheDocument();
  });

  it("uses a custom scope label when provided", () => {
    render(
      <EmptyInterviewsToggle
        emptyCount={4}
        allEmpty
        showEmpty={false}
        onToggle={vi.fn()}
        scope="test runs"
      />,
    );
    expect(
      screen.getByText("All 4 interviews in test runs have no transcript lines yet."),
    ).toBeInTheDocument();
  });
});
