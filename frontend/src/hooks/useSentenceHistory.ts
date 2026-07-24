import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/**
 * One event in a sentence's edit history. Pinned to
 * src/api/routers/edits.py::get_sentence_history's response shape.
 */
export interface SentenceHistoryEvent {
  event_type: string;
  version: number;
  occurred_at: string;
  actor: { actor_type: string; user_id: string };
  correlation_id: string;
  data: Record<string, unknown>;
}

export interface SentenceHistory {
  sentence_id: string;
  interview_id: string;
  sentence_index: number;
  current_version: number;
  current_text: string;
  is_edited: boolean;
  event_count: number;
  events: SentenceHistoryEvent[];
}

/**
 * Lazy edit-history fetch for one transcript line, from the EXISTING
 * `GET /edits/sentences/{interview_id}/{sentence_index}/history` endpoint
 * (src/api/routers/edits.py — not under /ui, still proxied via /api/*).
 *
 * Index semantics (pinned by reading edits.py + src/ingestion/orchestrator.py):
 * `sentence_index` derives the sentence's UUID via
 * `uuid5(NAMESPACE_DNS, f"{interview_id}:{sequence_order}")` — the exact
 * same derivation the ingestion orchestrator uses for `frag.sequence_order`.
 * So the transcript line's `sequence_order` IS the `sentence_index` this
 * endpoint expects.
 *
 * `enabled` gates the fetch — callers (LineDetailPanel) pass `isOpen` so the
 * request only fires once the detail panel is actually opened for a line.
 */
export function useSentenceHistory(
  interviewId: string,
  sequenceOrder: number,
  enabled: boolean,
) {
  return useQuery({
    queryKey: queryKeys.sentenceHistory(interviewId, sequenceOrder),
    queryFn: async () => {
      return (await apiGet(
        "/edits/sentences/{interview_id}/{sentence_index}/history",
        { params: { interview_id: interviewId, sentence_index: sequenceOrder } },
      )) as unknown as SentenceHistory;
    },
    enabled: enabled && Boolean(interviewId),
  });
}
