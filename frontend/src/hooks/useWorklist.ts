import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/**
 * Row/response shapes for `GET /review/worklist` — pinned to
 * src/api/routers/review.py::review_worklist. The endpoint returns a raw
 * dict, so the generated `schema.d.ts` types its 200 response as `unknown`
 * (see openapi-typescript output); these local shapes follow the same
 * precedent as useTranscript.ts/mutations.ts casting `unknown` to a typed
 * shape rather than hand-editing the generated schema.
 */
export interface WorklistLensItem {
  interview_id: string;
  item_id: string;
  node_type: string;
  lens: string;
  confidence: number;
  reason: "low_confidence" | "unresolved_reference";
}

export interface WorklistClaim {
  interview_id: string;
  claim_id: string;
  text: string;
  kind: string;
  confidence: number;
  reason: "low_confidence";
}

export interface WorklistEntityMergeSuggestion {
  surviving_canonical_id: string;
  merged_canonical_id: string;
  surfaces_a: string[];
  surfaces_b: string[];
  score: number;
  band: "auto" | "suggest";
}

export interface WorklistPersonLinkSuggestion {
  person_id: string;
  display_name: string;
  interview_id: string;
  speaker_id: string;
  speaker_display_name: string;
  reason: string;
}

export interface WorklistData {
  lens_items: WorklistLensItem[];
  claims: WorklistClaim[];
  entity_merge_suggestions: WorklistEntityMergeSuggestion[];
  person_link_suggestions: WorklistPersonLinkSuggestion[];
  /** May contain "embedding_unavailable" — degrades suggestion quality. */
  flags: string[];
}

/**
 * The project's review worklist (M5.0 Task 8): low-confidence lens items and
 * claims, plus entity-merge and person-link suggestions. Relies on the
 * endpoint's defaults for `threshold`/`limit`/`offset`.
 */
export function useWorklist(projectId: string) {
  return useQuery({
    queryKey: queryKeys.worklist(projectId),
    queryFn: async () => {
      return (await apiGet("/review/worklist", {
        query: { project_id: projectId },
      })) as unknown as WorklistData;
    },
    enabled: Boolean(projectId),
  });
}
