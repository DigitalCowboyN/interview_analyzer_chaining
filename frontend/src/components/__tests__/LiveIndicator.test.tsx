import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveIndicator } from "@/components/LiveIndicator";

describe("LiveIndicator", () => {
  it("announces live updates are on when status is live", () => {
    render(<LiveIndicator status="live" />);
    expect(screen.getByText(/live updates on/i)).toBeInTheDocument();
  });

  it("announces live updates are off when status is offline", () => {
    render(<LiveIndicator status="offline" />);
    expect(screen.getByText(/live updates off/i)).toBeInTheDocument();
  });

  it("announces live updates are off when status is idle", () => {
    render(<LiveIndicator status="idle" />);
    expect(screen.getByText(/live updates off/i)).toBeInTheDocument();
  });

  it("is an aria-live region so status flips are announced accessibly — NOT role=\"status\" (that role is StateGate's loading indicator on the same page, and a second one would make getByRole('status') ambiguous)", () => {
    render(<LiveIndicator status="live" />);
    expect(screen.getByText(/live updates on/i).closest("[aria-live]")).toHaveAttribute(
      "aria-live",
      "polite",
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
