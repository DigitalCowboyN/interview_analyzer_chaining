import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProjectCard } from "@/components/ProjectCard";

describe("ProjectCard", () => {
  it("shows the display name and count, linking to the project", () => {
    render(<ProjectCard href="/projects/samples" name="Samples" interviewCount={4} />);
    const link = screen.getByRole("link", { name: /Samples/ });
    expect(link).toHaveAttribute("href", "/projects/samples");
    expect(link).toHaveTextContent("4 interviews");
  });

  it("singularizes one interview", () => {
    render(<ProjectCard href="/x" name="X" interviewCount={1} />);
    expect(screen.getByRole("link")).toHaveTextContent("1 interview");
  });
});
