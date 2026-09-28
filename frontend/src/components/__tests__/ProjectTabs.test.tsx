import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { usePathname } from "next/navigation";
import { ProjectTabs } from "@/components/ProjectTabs";

vi.mock("next/navigation", () => ({ usePathname: vi.fn() }));

describe("ProjectTabs", () => {
  it("links every tab and marks the active one", () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/personas");
    render(<ProjectTabs projectId="samples" />);
    expect(screen.getByRole("link", { name: "Interviews" })).toHaveAttribute("href", "/projects/samples");
    expect(screen.getByRole("link", { name: "People" })).toHaveAttribute("href", "/projects/samples/people");
    expect(screen.getByRole("link", { name: "Personas" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Interviews" })).not.toHaveAttribute("aria-current");
  });

  it("treats interview pages as the Interviews tab", () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/interviews/i1");
    render(<ProjectTabs projectId="samples" />);
    expect(screen.getByRole("link", { name: "Interviews" })).toHaveAttribute("aria-current", "page");
  });
});
