import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useParams, usePathname, useRouter } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { IdentityProvider } from "@/identity/IdentityProvider";
import { useProjects } from "@/hooks/useProjects";

vi.mock("next/navigation", () => ({ useParams: vi.fn(), usePathname: vi.fn(), useRouter: vi.fn() }));
vi.mock("@/hooks/useProjects", () => ({ useProjects: vi.fn() }));

beforeEach(() => {
  vi.mocked(useRouter).mockReturnValue({ push: vi.fn() } as never);
  vi.mocked(usePathname).mockReturnValue("/projects/samples");
  vi.mocked(useParams).mockReturnValue({ projectId: "samples" });
  vi.mocked(useProjects).mockReturnValue({ data: [], isLoading: false } as never);
});

describe("AppHeader", () => {
  it("renders the brand link to the projects landing page, the project switcher, and the identity switcher", () => {
    render(
      <IdentityProvider>
        <AppHeader />
      </IdentityProvider>,
    );

    expect(screen.getByRole("link", { name: "Interview Analyzer" })).toHaveAttribute(
      "href",
      "/",
    );
    expect(screen.getByRole("combobox", { name: "Project" })).toBeInTheDocument();
    expect(screen.getByLabelText("User")).toBeInTheDocument();
  });
});
