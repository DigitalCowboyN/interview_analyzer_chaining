import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterviewRow } from "@/components/InterviewRow";

const base = {
  interview_id: "i1",
  title: "Weekly Planning Sync",
  created_at: "2026-09-27T10:12:13.623624Z",
  fragment_count: 81,
  participants: ["Maya Chen", "Priya Nandan"],
  insight_counts: { ActionItem: 18, Decision: 22, Trait: 3 },
};

describe("InterviewRow", () => {
  it("links to the interview and shows title, date, participants, lines", () => {
    render(<InterviewRow href="/projects/s/interviews/i1" interview={base} />);
    const link = screen.getByRole("link", { name: /Weekly Planning Sync/ });
    expect(link).toHaveAttribute("href", "/projects/s/interviews/i1");
    expect(link).toHaveTextContent("Sep 27, 2026");
    expect(link).toHaveTextContent("Maya Chen, Priya Nandan");
    expect(link).toHaveTextContent("81 lines");
  });

  it("renders insight chips in canonical order, unknown types last", () => {
    render(<InterviewRow href="/x" interview={base} />);
    const chips = screen.getAllByTestId("insight-chip").map((c) => c.textContent);
    expect(chips).toEqual(["22 decisions", "18 actions", "3 Trait"]);
  });

  it("omits chips when there are no insights", () => {
    render(<InterviewRow href="/x" interview={{ ...base, insight_counts: {} }} />);
    expect(screen.queryAllByTestId("insight-chip")).toHaveLength(0);
  });
});
