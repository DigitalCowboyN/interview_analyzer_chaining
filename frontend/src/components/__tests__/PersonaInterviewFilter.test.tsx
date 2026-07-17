import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PersonaInterviewFilter } from "@/components/PersonaInterviewFilter";

const INTERVIEWS = [
  { interview_id: "i1", title: "Kickoff call", created_at: "2026-01-01", fragment_count: 10 },
  { interview_id: "i2", title: "Follow-up", created_at: "2026-01-02", fragment_count: 5 },
];

const PERSONAS = [
  {
    person_id: "p1",
    display_name: "Jane Doe",
    trait_count: 1,
    goal_count: 1,
    pain_point_count: 0,
    quote_count: 0,
    representative_quote: null,
    interview_ids: ["i1"],
  },
  {
    person_id: "p2",
    display_name: "John Roe",
    trait_count: 1,
    goal_count: 1,
    pain_point_count: 0,
    quote_count: 0,
    representative_quote: null,
    interview_ids: ["i2"],
  },
];

describe("PersonaInterviewFilter", () => {
  it("shows all personas when no interview is selected", () => {
    render(
      <PersonaInterviewFilter
        projectId="proj1"
        interviews={INTERVIEWS}
        personas={PERSONAS}
        selectedInterviewId=""
        onSelectInterview={vi.fn()}
      />,
    );

    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.getByText("John Roe")).toBeInTheDocument();
  });

  it("filters persona profiles present in the selected interview", () => {
    render(
      <PersonaInterviewFilter
        projectId="proj1"
        interviews={INTERVIEWS}
        personas={PERSONAS}
        selectedInterviewId="i1"
        onSelectInterview={vi.fn()}
      />,
    );

    expect(screen.getByText("Jane Doe")).toBeInTheDocument();
    expect(screen.queryByText("John Roe")).not.toBeInTheDocument();
  });

  it("calls onSelectInterview when the selector changes", () => {
    const onSelectInterview = vi.fn();
    render(
      <PersonaInterviewFilter
        projectId="proj1"
        interviews={INTERVIEWS}
        personas={PERSONAS}
        selectedInterviewId=""
        onSelectInterview={onSelectInterview}
      />,
    );

    fireEvent.change(screen.getByLabelText("Filter by interview"), {
      target: { value: "i2" },
    });

    expect(onSelectInterview).toHaveBeenCalledWith("i2");
  });

  it("shows an empty message when no personas match the selected interview", () => {
    render(
      <PersonaInterviewFilter
        projectId="proj1"
        interviews={INTERVIEWS}
        personas={[]}
        selectedInterviewId="i1"
        onSelectInterview={vi.fn()}
      />,
    );

    expect(
      screen.getByText("No persona profiles for this interview."),
    ).toBeInTheDocument();
  });
});
