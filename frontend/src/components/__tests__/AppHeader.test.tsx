import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppHeader } from "@/components/AppHeader";
import { IdentityProvider } from "@/identity/IdentityProvider";

describe("AppHeader", () => {
  it("renders the brand link to the projects landing page and the identity switcher", () => {
    render(
      <IdentityProvider>
        <AppHeader />
      </IdentityProvider>,
    );

    expect(screen.getByRole("link", { name: "Interview Analyzer" })).toHaveAttribute(
      "href",
      "/",
    );
    expect(screen.getByLabelText("User")).toBeInTheDocument();
  });
});
