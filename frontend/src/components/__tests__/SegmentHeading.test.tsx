import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SegmentHeading } from "@/components/SegmentHeading";

describe("SegmentHeading", () => {
  it("renders the segment topic", () => {
    render(<SegmentHeading topic="Onboarding friction" />);
    expect(
      screen.getByRole("heading", { name: "Onboarding friction" }),
    ).toBeInTheDocument();
  });

  it("falls back to a placeholder when topic is null", () => {
    render(<SegmentHeading topic={null} />);
    expect(
      screen.getByRole("heading", { name: "Untitled segment" }),
    ).toBeInTheDocument();
  });
});
