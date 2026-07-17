import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PersonCardGrid } from "@/components/PersonCardGrid";

describe("PersonCardGrid", () => {
  it("renders card content (name, speaker/interview counts) and navigates to the person core view", () => {
    render(
      <PersonCardGrid
        projectId="proj1"
        persons={[
          { person_id: "p1", display_name: "Jane Doe", speaker_count: 2, interview_count: 3 },
        ]}
      />,
    );

    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();

    // Separate-route-tree assertion: person cards navigate under /gallery/persons/...
    expect(screen.getByRole("link", { name: /Jane Doe/ })).toHaveAttribute(
      "href",
      "/gallery/persons/proj1/p1",
    );
  });
});
