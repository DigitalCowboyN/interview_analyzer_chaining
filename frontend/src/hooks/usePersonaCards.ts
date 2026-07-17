import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One row of `GET /ui/projects/{project_id}/personas` — pinned to
 * src/api/routers/ui.py::list_personas / src/ui/reader.py::persona_card_rows. */
export interface PersonaCardData {
  person_id: string;
  display_name: string;
  trait_count: number;
  goal_count: number;
  pain_point_count: number;
  quote_count: number;
  representative_quote: string | null;
  interview_ids: string[];
}

interface PersonaCardsResponse {
  personas: PersonaCardData[];
}

/** Persona-profile cards for a project — the gallery's persona discovery grid. */
export function usePersonaCards(projectId: string) {
  return useQuery({
    queryKey: queryKeys.personas(projectId),
    queryFn: async () => {
      const data = (await apiGet("/ui/projects/{project_id}/personas", {
        params: { project_id: projectId },
      })) as unknown as PersonaCardsResponse;
      return data.personas;
    },
    enabled: Boolean(projectId),
  });
}
