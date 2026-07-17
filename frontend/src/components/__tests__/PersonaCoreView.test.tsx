import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PersonaCoreView } from "@/components/PersonaCoreView";

describe("PersonaCoreView", () => {
  it("renders dimension-grouped items with per-interview provenance chips", () => {
    render(
      <PersonaCoreView
        dimensions={{
          traits: [
            {
              item_id: "t1",
              text: "Detail-oriented",
              confidence: 0.9,
              interview_id: "i1",
              interview_title: "Kickoff call",
            },
          ],
          goals: [
            {
              item_id: "g1",
              text: "Ship faster",
              confidence: 0.75,
              interview_id: "i2",
              interview_title: "Follow-up",
            },
          ],
          pain_points: [],
          notable_quotes: [],
        }}
      />,
    );

    expect(screen.getByText("Traits (1)")).toBeInTheDocument();
    expect(screen.getByText("Detail-oriented")).toBeInTheDocument();
    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
    expect(screen.getByText("90%")).toBeInTheDocument();

    expect(screen.getByText("Goals (1)")).toBeInTheDocument();
    expect(screen.getByText("Ship faster")).toBeInTheDocument();
    expect(screen.getByText("Follow-up")).toBeInTheDocument();

    expect(screen.getByText("Pain points (0)")).toBeInTheDocument();
    expect(screen.getByText("Notable quotes (0)")).toBeInTheDocument();
    expect(screen.getAllByText("None recorded.")).toHaveLength(2);
  });
});
