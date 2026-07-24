import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One row of `GET /ui/projects/{project_id}/persons` — pinned to
 * src/api/routers/ui.py::list_persons. */
export interface PersonSummary {
  person_id: string;
  display_name: string;
  speaker_count: number;
  interview_count: number;
}

interface PersonsResponse {
  persons: PersonSummary[];
}

/** The project's persons — backs the manual speaker→person picker (Task 6). */
export function usePersons(projectId: string) {
  return useQuery({
    queryKey: queryKeys.persons(projectId),
    queryFn: async () => {
      const data = (await apiGet("/ui/projects/{project_id}/persons", {
        params: { project_id: projectId },
      })) as PersonsResponse;
      return data.persons;
    },
    enabled: Boolean(projectId),
  });
}

/**
 * Compute-only person-id derivation for the create-new-person flow
 * (`GET /ui/projects/{project_id}/person-id?display_name=…`). NOT a
 * `useQuery` — the picker calls this imperatively, once, when the user
 * commits a new name, then passes the derived id + display_name straight
 * through to the link call. The frontend never derives the id itself
 * (loose-coupling requirement, spec's Global Constraints) — this hook
 * exists purely to keep the derivation call typed and in one place.
 */
export function usePersonId() {
  return {
    derivePersonId: async (projectId: string, displayName: string): Promise<string> => {
      const data = (await apiGet("/ui/projects/{project_id}/person-id", {
        params: { project_id: projectId },
        query: { display_name: displayName },
      })) as { person_id: string };
      return data.person_id;
    },
  };
}
