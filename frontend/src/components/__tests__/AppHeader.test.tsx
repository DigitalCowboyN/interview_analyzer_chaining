import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppHeader } from "@/components/AppHeader";
import { IdentityProvider } from "@/identity/IdentityProvider";

vi.mock("next/navigation", () => ({
  usePathname: () => "/workbench",
}));

describe("AppHeader", () => {
  it("renders Workbench and Gallery nav links and the identity switcher", () => {
    render(
      <IdentityProvider>
        <AppHeader />
      </IdentityProvider>,
    );

    expect(screen.getByRole("link", { name: "Workbench" })).toHaveAttribute(
      "href",
      "/workbench",
    );
    expect(screen.getByRole("link", { name: "Gallery" })).toHaveAttribute(
      "href",
      "/gallery",
    );
    expect(screen.getByLabelText("User")).toBeInTheDocument();
  });
});
