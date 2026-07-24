import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { PersonCoreView } from "@/components/PersonCoreView";

describe("PersonCoreView", () => {
  it("renders identity facts (linked speakers per interview)", () => {
    render(
      <PersonCoreView
        projectId="proj1"
        personId="p1"
        links={[
          {
            interview_id: "i1",
            interview_title: "Kickoff call",
            speaker_id: "s1",
            speaker_display_name: "Speaker A",
          },
        ]}
        contributesToPersona={false}
      />,
    );

    expect(screen.getByText("Linked speakers (1)")).toBeInTheDocument();
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
  });

  it("links to the persona core view (navigation between views, never an embed) when contributing", () => {
    render(
      <PersonCoreView projectId="proj1" personId="p1" links={[]} contributesToPersona={true} />,
    );

    expect(screen.getByRole("link", { name: /View persona profile/ })).toHaveAttribute(
      "href",
      "/gallery/personas/proj1/p1",
    );
  });

  it("shows a quiet no-persona-profile note when not contributing", () => {
    render(
      <PersonCoreView projectId="proj1" personId="p1" links={[]} contributesToPersona={false} />,
    );

    expect(
      screen.getByText("No persona profile yet for this person."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /View persona profile/ })).not.toBeInTheDocument();
  });
});
