import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";
import type { InterviewSummary } from "@/hooks/useInterviews";

/** Row of `GET /ui/test-runs/interviews` — pinned to src/api/routers/ui.py::list_test_run_interviews. */
export type TestRunInterview = InterviewSummary & { project_id: string; suite: string };

export function useTestRunInterviews() {
  return useQuery({
    queryKey: queryKeys.testRunInterviews(),
    queryFn: async () => {
      const data = (await apiGet("/ui/test-runs/interviews")) as { interviews: TestRunInterview[] };
      return data.interviews;
    },
  });
}
