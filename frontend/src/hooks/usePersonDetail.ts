import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One linked speaker row — pinned to
 * src/ui/reader.py::person_detail_rows. */
export interface PersonLink {
  interview_id: string;
  interview_title: string;
  speaker_id: string;
  speaker_display_name: string;
}

/** `GET /ui/persons/{project_id}/{person_id}` — pinned to
 * src/api/routers/ui.py::get_person. `contributes_to_persona` is a loose
 * link flag (never an embed) to the person's persona profile. */
export interface PersonDetailData {
  person_id: string;
  display_name: string | null;
  links: PersonLink[];
  contributes_to_persona: boolean;
}

/** Person core view: identity facts (linked speakers per interview). */
export function usePersonDetail(projectId: string, personId: string) {
  return useQuery({
    queryKey: queryKeys.person(projectId, personId),
    queryFn: async () => {
      return (await apiGet("/ui/persons/{project_id}/{person_id}", {
        params: { project_id: projectId, person_id: personId },
      })) as unknown as PersonDetailData;
    },
    enabled: Boolean(projectId) && Boolean(personId),
  });
}
