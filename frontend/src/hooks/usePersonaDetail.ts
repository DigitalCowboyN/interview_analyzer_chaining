import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** One dimension item with per-interview provenance — pinned to
 * src/api/routers/ui.py::_shape_dimension_item. */
export interface PersonaDimensionItem {
  item_id: string;
  text: string;
  confidence: number;
  interview_id: string;
  interview_title: string;
}

export interface PersonaDimensions {
  traits: PersonaDimensionItem[];
  goals: PersonaDimensionItem[];
  pain_points: PersonaDimensionItem[];
  notable_quotes: PersonaDimensionItem[];
}

/** `GET /ui/personas/{project_id}/{person_id}` — pinned to
 * src/api/routers/ui.py::get_persona. */
export interface PersonaDetailData {
  person_id: string;
  display_name: string | null;
  dimensions: PersonaDimensions;
}

/** Persona core view: dimension-grouped items with per-interview provenance. */
export function usePersonaDetail(projectId: string, personId: string) {
  return useQuery({
    queryKey: queryKeys.persona(projectId, personId),
    queryFn: async () => {
      return (await apiGet("/ui/personas/{project_id}/{person_id}", {
        params: { project_id: projectId, person_id: personId },
      })) as unknown as PersonaDetailData;
    },
    enabled: Boolean(projectId) && Boolean(personId),
  });
}
