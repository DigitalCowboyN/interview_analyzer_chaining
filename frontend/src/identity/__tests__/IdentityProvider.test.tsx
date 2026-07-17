import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { IdentityProvider, useIdentity } from "@/identity/IdentityProvider";
import { IdentitySwitcher } from "@/identity/IdentitySwitcher";

function Harness() {
  const { userId } = useIdentity();
  return <div data-testid="current-user">{userId}</div>;
}

describe("IdentityProvider", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("defaults to 'dev'", async () => {
    render(
      <IdentityProvider>
        <Harness />
      </IdentityProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("current-user")).toHaveTextContent("dev"),
    );
  });

  it("persists the selected user id to localStorage and reflects it on remount", async () => {
    const user = userEvent.setup();
    const { unmount } = render(
      <IdentityProvider>
        <IdentitySwitcher />
        <Harness />
      </IdentityProvider>,
    );

    await user.selectOptions(screen.getByLabelText("User"), "reviewer");

    await waitFor(() =>
      expect(screen.getByTestId("current-user")).toHaveTextContent("reviewer"),
    );
    expect(window.localStorage.getItem("interview-analyzer:user-id")).toBe(
      "reviewer",
    );

    unmount();

    render(
      <IdentityProvider>
        <Harness />
      </IdentityProvider>,
    );
    await waitFor(() =>
      expect(screen.getByTestId("current-user")).toHaveTextContent("reviewer"),
    );
  });

  it("supports free-text custom user ids", async () => {
    const user = userEvent.setup();
    render(
      <IdentityProvider>
        <IdentitySwitcher />
        <Harness />
      </IdentityProvider>,
    );

    await user.selectOptions(screen.getByLabelText("User"), "Custom…");
    await user.type(screen.getByLabelText("Custom user id"), "alice");
    await user.click(screen.getByRole("button", { name: "Set" }));

    await waitFor(() =>
      expect(screen.getByTestId("current-user")).toHaveTextContent("alice"),
    );
    expect(window.localStorage.getItem("interview-analyzer:user-id")).toBe(
      "alice",
    );
  });
});
