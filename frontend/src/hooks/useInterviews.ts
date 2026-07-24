import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One row of `GET /ui/projects/{project_id}/interviews` — pinned to
 * src/api/routers/ui.py::list_interviews. */
export interface InterviewSummary {
  interview_id: string;
  title: string;
  created_at: string;
  fragment_count: number;
}

interface InterviewsResponse {
  interviews: InterviewSummary[];
}

/** Interviews for a project, keyed by project id (workbench drill-down). */
export function useInterviews(projectId: string) {
  return useQuery({
    queryKey: queryKeys.interviews(projectId),
    queryFn: async () => {
      const data = (await apiGet("/ui/projects/{project_id}/interviews", {
        params: { project_id: projectId },
      })) as InterviewsResponse;
      return data.interviews;
    },
    enabled: Boolean(projectId),
  });
}
