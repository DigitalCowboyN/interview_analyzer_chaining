import { TEST_RUNS_ID } from "@/lib/routes";

interface Nameable {
  project_id: string;
  kind?: "real" | "test";
  suite?: string | null;
}

/** Human label for a project: "Samples", "ui-smoke · 150548b2", "Test runs". */
export function displayProjectName(project: Nameable): string {
  if (project.project_id === TEST_RUNS_ID) return "Test runs";
  if (project.kind === "test" && project.suite) {
    const rest = project.project_id.slice(project.suite.length + 1);
    return `${project.suite} · ${rest.split("-")[0] || project.project_id}`;
  }
  if (project.kind === "real") {
    return project.project_id
      .split(/[-_]+/)
      .filter(Boolean)
      .map((word) => word[0].toUpperCase() + word.slice(1))
      .join(" ");
  }
  return project.project_id;
}
