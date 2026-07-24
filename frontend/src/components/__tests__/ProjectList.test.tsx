import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProjectList } from "@/components/ProjectList";

describe("ProjectList", () => {
  it("renders each project with its interview count and links to its interviews screen", () => {
    render(
      <ProjectList
        projects={[
          { project_id: "p1", interview_count: 3 },
          { project_id: "p2", interview_count: 1 },
        ]}
      />,
    );

    expect(screen.getByText("p1")).toBeInTheDocument();
    expect(screen.getByText("3 interviews")).toBeInTheDocument();
    expect(screen.getByText("1 interview")).toBeInTheDocument();

    expect(screen.getByRole("link", { name: /p1/ })).toHaveAttribute(
      "href",
      "/workbench/p1",
    );
    expect(screen.getByRole("link", { name: /p2/ })).toHaveAttribute(
      "href",
      "/workbench/p2",
    );
  });
});
