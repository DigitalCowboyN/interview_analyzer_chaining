import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";

/** Pinned to src/api/routers/ui.py::get_transcript / src/ui/reader.py shaping. */
export interface TranscriptSpeaker {
  speaker_id: string;
  display_name: string;
}

export interface TranscriptPerson {
  person_id: string;
  display_name: string;
}

export interface TranscriptSegment {
  segment_id: string;
  topic: string | null;
}

export interface TranscriptEntity {
  surface: string;
  entity_type: string;
}

export interface TranscriptLensItem {
  item_id: string;
  lens: string;
  node_type: string;
  text: string;
  confidence: number;
  human_locked: boolean;
}

export interface TranscriptLineData {
  fragment_id: string;
  sequence_order: number;
  text: string;
  speaker: TranscriptSpeaker | null;
  person: TranscriptPerson | null;
  utterance_id: string | null;
  segment: TranscriptSegment | null;
  entities: TranscriptEntity[];
  lens_items: TranscriptLensItem[];
  edited: boolean;
}

/**
 * Front-matter metadata for the interview. KNOWN BACKEND GAP (M5.0 Task 1):
 * the Interview node carries no metadata property yet, so this is always
 * `{}` until a projection handler starts writing front matter onto the
 * graph. The MetadataPanel shows a quiet "no metadata available" state
 * when this is empty rather than treating it as an error.
 */
export type TranscriptMetadata = Record<string, unknown>;

export interface TranscriptData {
  interview_id: string;
  title: string;
  metadata: TranscriptMetadata;
  lines: TranscriptLineData[];
}

/** Full transcript (lines + metadata) for one interview — the workbench core display. */
export function useTranscript(interviewId: string) {
  return useQuery({
    queryKey: queryKeys.transcript(interviewId),
    queryFn: async () => {
      return (await apiGet("/ui/interviews/{interview_id}/transcript", {
        params: { interview_id: interviewId },
      })) as unknown as TranscriptData;
    },
    enabled: Boolean(interviewId),
  });
}
