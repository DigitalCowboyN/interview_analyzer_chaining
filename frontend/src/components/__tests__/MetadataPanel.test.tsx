import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetadataPanel } from "@/components/MetadataPanel";

describe("MetadataPanel", () => {
  it("renders the title and a quiet empty state when metadata is {} (known backend gap)", () => {
    render(<MetadataPanel title="Kickoff call" metadata={{}} />);
    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
    expect(screen.getByText("No metadata available.")).toBeInTheDocument();
  });

  it("renders front-matter fields when metadata is populated", () => {
    render(
      <MetadataPanel
        title="Kickoff call"
        metadata={{ interviewer: "Jane Doe", date: "2026-01-01" }}
      />,
    );
    expect(screen.getByText("interviewer")).toBeInTheDocument();
    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText("date")).toBeInTheDocument();
    expect(screen.queryByText("No metadata available.")).not.toBeInTheDocument();
  });
});
