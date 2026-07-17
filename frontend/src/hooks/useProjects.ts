import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One row of `GET /ui/projects` — pinned to src/api/routers/ui.py::list_projects. */
export interface ProjectSummary {
  project_id: string;
  interview_count: number;
}

interface ProjectsResponse {
  projects: ProjectSummary[];
}

/** Projects list for the workbench nav root. */
export function useProjects() {
  return useQuery({
    queryKey: queryKeys.projects(),
    queryFn: async () => {
      const data = (await apiGet("/ui/projects")) as ProjectsResponse;
      return data.projects;
    },
  });
}
