import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Breadcrumbs } from "@/components/Breadcrumbs";

describe("Breadcrumbs", () => {
  it("renders the Workbench / project / interview trail with links on all but the last crumb", () => {
    render(
      <Breadcrumbs
        items={[
          { label: "Workbench", href: "/workbench" },
          { label: "p1", href: "/workbench/p1" },
          { label: "Kickoff call" },
        ]}
      />,
    );

    expect(screen.getByRole("link", { name: "Workbench" })).toHaveAttribute(
      "href",
      "/workbench",
    );
    expect(screen.getByRole("link", { name: "p1" })).toHaveAttribute(
      "href",
      "/workbench/p1",
    );
    expect(screen.getByText("Kickoff call")).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Kickoff call" }),
    ).not.toBeInTheDocument();
  });
});
