import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterviewList } from "@/components/InterviewList";

describe("InterviewList", () => {
  it("renders each interview with title, created date, fragment count, and navigates to the transcript route", () => {
    render(
      <InterviewList
        projectId="p1"
        interviews={[
          {
            interview_id: "i1",
            title: "Kickoff call",
            created_at: "2026-01-01T00:00:00Z",
            fragment_count: 42,
          },
        ]}
      />,
    );

    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
    expect(screen.getByText("2026-01-01T00:00:00Z")).toBeInTheDocument();
    expect(screen.getByText("42 fragments")).toBeInTheDocument();

    expect(screen.getByRole("link", { name: /Kickoff call/ })).toHaveAttribute(
      "href",
      "/workbench/p1/i1",
    );
  });
});
