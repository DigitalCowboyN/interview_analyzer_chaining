import Link from "next/link";
import type { ProjectSummary } from "@/hooks/useProjects";

/** Renders project rows (id + interview count); each navigates to its interviews screen. */
export function ProjectList({ projects }: { projects: ProjectSummary[] }) {
  return (
    <ul className="divide-y divide-border">
      {projects.map((project) => (
        <li key={project.project_id}>
          <Link
            href={`/workbench/${encodeURIComponent(project.project_id)}`}
            className="flex items-center justify-between py-3 hover:bg-surface-raised"
          >
            <span className="font-medium text-fg">
              {project.project_id}
            </span>
            <span className="text-sm text-fg-muted">
              {project.interview_count}{" "}
              {project.interview_count === 1 ? "interview" : "interviews"}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
