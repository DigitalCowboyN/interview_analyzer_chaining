import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PersonaCardGrid } from "@/components/PersonaCardGrid";

describe("PersonaCardGrid", () => {
  it("renders card content (name, dimension counts, representative quote) and navigates to the persona core view", () => {
    render(
      <PersonaCardGrid
        projectId="proj1"
        personas={[
          {
            person_id: "p1",
            display_name: "Jane Doe",
            trait_count: 2,
            goal_count: 1,
            pain_point_count: 3,
            quote_count: 4,
            representative_quote: "I just want it to work.",
            interview_ids: ["i1"],
          },
        ]}
      />,
    );

    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText(/I just want it to work\./)).toBeInTheDocument();

    // Separate-route-tree assertion: persona cards navigate under /gallery/personas/...
    expect(screen.getByRole("link", { name: /Jane Doe/ })).toHaveAttribute(
      "href",
      "/gallery/personas/proj1/p1",
    );
  });

  it("renders no representative quote block when null", () => {
    const { container } = render(
      <PersonaCardGrid
        projectId="proj1"
        personas={[
          {
            person_id: "p2",
            display_name: "No Quote",
            trait_count: 0,
            goal_count: 0,
            pain_point_count: 0,
            quote_count: 0,
            representative_quote: null,
            interview_ids: [],
          },
        ]}
      />,
    );

    expect(container.querySelector("p.italic")).not.toBeInTheDocument();
  });
});
