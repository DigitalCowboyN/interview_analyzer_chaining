import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useParams, usePathname, useRouter } from "next/navigation";
import { ProjectSwitcher } from "@/components/ProjectSwitcher";
import { useProjects } from "@/hooks/useProjects";

vi.mock("next/navigation", () => ({ useParams: vi.fn(), usePathname: vi.fn(), useRouter: vi.fn() }));
vi.mock("@/hooks/useProjects", () => ({ useProjects: vi.fn() }));

const push = vi.fn();
const projects = [
  { project_id: "samples", interview_count: 4, kind: "real", suite: null },
  { project_id: "real-interviews", interview_count: 1, kind: "real", suite: null },
  { project_id: "smoke-1", interview_count: 1, kind: "test", suite: "smoke" },
];

beforeEach(() => {
  push.mockReset();
  vi.mocked(useRouter).mockReturnValue({ push } as never);
  vi.mocked(usePathname).mockReturnValue("/projects/samples");
  vi.mocked(useParams).mockReturnValue({ projectId: "samples" });
  vi.mocked(useProjects).mockReturnValue({ data: projects, isLoading: false } as never);
});

describe("ProjectSwitcher", () => {
  it("selects the project from the URL and lists real projects plus Test runs", () => {
    render(<ProjectSwitcher />);
    const select = screen.getByRole("combobox", { name: "Project" });
    expect(select).toHaveValue("samples");
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels).toEqual(["All projects", "Samples", "Real Interviews", "Test runs"]);
  });

  it("navigates to the chosen project", async () => {
    render(<ProjectSwitcher />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Project" }), "real-interviews");
    expect(push).toHaveBeenCalledWith("/projects/real-interviews");
  });

  it("navigates home for All projects and to the bucket for Test runs", async () => {
    render(<ProjectSwitcher />);
    const select = screen.getByRole("combobox", { name: "Project" });
    await userEvent.selectOptions(select, "test-runs");
    expect(push).toHaveBeenLastCalledWith("/projects/test-runs");
    await userEvent.selectOptions(select, "");
    expect(push).toHaveBeenLastCalledWith("/");
  });

  it("maps a test project's own URL to the Test runs entry", () => {
    vi.mocked(useParams).mockReturnValue({ projectId: "smoke-1" });
    render(<ProjectSwitcher />);
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("test-runs");
  });

  it("shows an unlisted project id as its own option instead of crashing", () => {
    vi.mocked(useParams).mockReturnValue({ projectId: "ghost" });
    render(<ProjectSwitcher />);
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("ghost");
    expect(screen.getByRole("option", { name: "ghost" })).toBeInTheDocument();
  });

  it("keeps the current tab when switching projects", async () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/personas/p-1");
    render(<ProjectSwitcher />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Project" }), "real-interviews");
    expect(push).toHaveBeenCalledWith("/projects/real-interviews/personas");
  });

  it("drops an interview id and goes to the project home when switching", async () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/interviews/i-1");
    render(<ProjectSwitcher />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Project" }), "real-interviews");
    expect(push).toHaveBeenCalledWith("/projects/real-interviews");
  });
});
