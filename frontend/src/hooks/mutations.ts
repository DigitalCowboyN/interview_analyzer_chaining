import { useCallback, useState } from "react";
import type { QueryClient, QueryKey } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/api/client";

/**
 * The correction-intent pattern (M5.0 Task 5). Every write in the workbench
 * is a command, not a direct state change — "commands are intents" (binding
 * doctrine, see docs/superpowers/plans/2026-07-17-m50-ui-scaffolding.md). This
 * module is the ONE wrapper; Tasks 6 and 8 reuse it verbatim for their own
 * flows (speaker linking, worklist accepts).
 *
 * Lifecycle:
 *   fire → "pending" (optimistic marker on the caller's element) →
 *   202 → bounded confirm-poll (every 2s, max 10 tries) against a caller-
 *     supplied predicate over a freshly-refetched query →
 *     - predicate true  → invalidate + "settled"
 *     - bounds exceeded → "timeout" (NOT an error — "still processing,
 *       check back later")
 *   409 → revert optimistic state, surface the server's `detail` message
 *   network error (fetch throws, or any other non-2xx status) → revert,
 *     generic notice
 *
 * The frontend never encodes Neo4j/ESDB/projection language in copy — only
 * "processing" — per the spec's loose-coupling section.
 */

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_ATTEMPTS = 10;

export type CorrectionStatus = "idle" | "pending" | "settled" | "timeout" | "reverted";

export interface CorrectionNotice {
  /** "timeout" for the bounded-poll exhaustion case (not an error tone);
   * "conflict" for 409s (server detail); "network" for transport/other failures. */
  kind: "timeout" | "conflict" | "network";
  message: string;
}

export interface RunCorrectionIntentArgs {
  queryClient: QueryClient;
  /** Fires the mutation. Must resolve to the raw Response so 202/409/other
   * status codes can be distinguished (use `apiFetch`, not apiPost/apiDelete). */
  request: () => Promise<Response>;
  /** Query key to refetch while polling and invalidate on settle. */
  queryKey: QueryKey;
  /** Predicate over the freshly-refetched query data: has the change landed? */
  isReflected: (data: unknown) => boolean;
}

export interface CorrectionOutcome {
  status: "settled" | "timeout" | "reverted";
  notice?: CorrectionNotice;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollUntilReflected(
  queryClient: QueryClient,
  queryKey: QueryKey,
  isReflected: (data: unknown) => boolean,
): Promise<boolean> {
  for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt += 1) {
    await sleep(POLL_INTERVAL_MS);
    const data = await queryClient.refetchQueries({ queryKey, exact: true }).then(
      () => queryClient.getQueryData(queryKey),
    );
    if (isReflected(data)) return true;
  }
  return false;
}

async function extractDetail(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") return data.detail;
  } catch {
    // fall through to a generic message
  }
  return `Request failed (${response.status}).`;
}

/**
 * Framework-agnostic core of the intent pattern — no React state, callable
 * directly from tests with fake timers. Returns the terminal outcome; never
 * throws (all failure paths are represented in the returned outcome).
 */
export async function runCorrectionIntent(
  args: RunCorrectionIntentArgs,
): Promise<CorrectionOutcome> {
  const { queryClient, request, queryKey, isReflected } = args;

  let response: Response;
  try {
    response = await request();
  } catch {
    return {
      status: "reverted",
      notice: {
        kind: "network",
        message: "Could not reach the server. Your change was not saved.",
      },
    };
  }

  if (response.status === 409) {
    const detail = await extractDetail(response);
    return { status: "reverted", notice: { kind: "conflict", message: detail } };
  }

  if (!response.ok) {
    const detail = await extractDetail(response);
    return { status: "reverted", notice: { kind: "network", message: detail } };
  }

  const reflected = await pollUntilReflected(queryClient, queryKey, isReflected);
  if (!reflected) {
    return {
      status: "timeout",
      notice: { kind: "timeout", message: "Still processing — check back later." },
    };
  }

  await queryClient.invalidateQueries({ queryKey });
  return { status: "settled" };
}

export interface UseCorrectionIntentOptions {
  queryKey: QueryKey;
}

export interface UseCorrectionIntentResult {
  status: CorrectionStatus;
  notice: CorrectionNotice | null;
  /** True while pending or actively polling — the caller's optimistic marker. */
  isPending: boolean;
  run: (
    request: () => Promise<Response>,
    isReflected: (data: unknown) => boolean,
  ) => Promise<CorrectionOutcome>;
  reset: () => void;
}

/**
 * React-facing wrapper: holds pending/notice state for one in-flight
 * correction against a fixed query key. `isReflected` is supplied per call
 * (it depends on the specific thing being corrected — which sentence, which
 * speaker, which segment — not just the flow). Per-flow hooks below fix that
 * predicate-building so components stay prop-driven: they render
 * `status`/`notice`, they never touch fetch.
 */
export function useCorrectionIntent(
  options: UseCorrectionIntentOptions,
): UseCorrectionIntentResult {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<CorrectionStatus>("idle");
  const [notice, setNotice] = useState<CorrectionNotice | null>(null);

  const run = useCallback(
    async (request: () => Promise<Response>, isReflected: (data: unknown) => boolean) => {
      setStatus("pending");
      setNotice(null);
      const outcome = await runCorrectionIntent({
        queryClient,
        request,
        queryKey: options.queryKey,
        isReflected,
      });
      setStatus(outcome.status);
      setNotice(outcome.notice ?? null);
      return outcome;
    },
    [queryClient, options.queryKey],
  );

  const reset = useCallback(() => {
    setStatus("idle");
    setNotice(null);
  }, []);

  return { status, notice, isPending: status === "pending", run, reset };
}

// --- Per-flow hooks (M5.0 Task 5) ---
//
// Each flow hook fixes the request-building and the reflected-predicate
// shape for one correction against the transcript read, so components only
// ever call e.g. `editText(sentenceIndex, text)`.

interface TranscriptLineShape {
  fragment_id: string;
  sequence_order: number;
  text: string;
  speaker: { speaker_id: string; display_name: string } | null;
  segment: { segment_id: string; topic: string | null } | null;
  lens_items: { item_id: string; human_locked: boolean }[];
  edited: boolean;
}

interface TranscriptShape {
  lines?: TranscriptLineShape[];
}

function findLine(
  data: unknown,
  match: (line: TranscriptLineShape) => boolean,
): TranscriptLineShape | undefined {
  const transcript = data as TranscriptShape | undefined;
  return transcript?.lines?.find(match);
}

/** Flow 1: transcript text edit. POST /edits/sentences/{interview_id}/{sentence_index}/edit */
export function useTextEditIntent(interviewId: string, transcriptQueryKey: QueryKey) {
  const intent = useCorrectionIntent({ queryKey: transcriptQueryKey });

  const editText = useCallback(
    (sentenceIndex: number, text: string, note?: string) =>
      intent.run(
        () =>
          apiFetch(`/edits/sentences/${interviewId}/${sentenceIndex}/edit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, editor_type: "human", ...(note ? { note } : {}) }),
          }),
        (data) =>
          Boolean(
            findLine(data, (line) => line.sequence_order === sentenceIndex && line.edited),
          ),
      ),
    [intent, interviewId],
  );

  return { ...intent, editText };
}

/** Flow 2a: speaker rename. POST /speakers/{interview_id}/{speaker_id}/rename */
export function useSpeakerRenameIntent(interviewId: string, transcriptQueryKey: QueryKey) {
  const intent = useCorrectionIntent({ queryKey: transcriptQueryKey });

  const renameSpeaker = useCallback(
    (speakerId: string, newDisplayName: string) =>
      intent.run(
        () =>
          apiFetch(`/speakers/${interviewId}/${speakerId}/rename`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ new_display_name: newDisplayName }),
          }),
        (data) =>
          Boolean(
            findLine(
              data,
              (line) =>
                line.speaker?.speaker_id === speakerId &&
                line.speaker?.display_name === newDisplayName,
            ),
          ),
      ),
    [intent, interviewId],
  );

  return { ...intent, renameSpeaker };
}

/** Flow 2b: fragment reattribute. POST /speakers/{interview_id}/fragments/{index}/reattribute */
export function useFragmentReattributeIntent(interviewId: string, transcriptQueryKey: QueryKey) {
  const intent = useCorrectionIntent({ queryKey: transcriptQueryKey });

  const reattributeFragment = useCallback(
    (fragmentIndex: number, newSpeakerId: string) =>
      intent.run(
        () =>
          apiFetch(`/speakers/${interviewId}/fragments/${fragmentIndex}/reattribute`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ new_speaker_id: newSpeakerId }),
          }),
        (data) =>
          Boolean(
            findLine(
              data,
              (line) =>
                line.sequence_order === fragmentIndex &&
                line.speaker?.speaker_id === newSpeakerId,
            ),
          ),
      ),
    [intent, interviewId],
  );

  return { ...intent, reattributeFragment };
}

/** Flow 3: segment remove. DELETE /segments/{interview_id}/{segment_id}?reason=… */
export function useSegmentRemoveIntent(interviewId: string, transcriptQueryKey: QueryKey) {
  const intent = useCorrectionIntent({ queryKey: transcriptQueryKey });

  const removeSegment = useCallback(
    (segmentId: string, reason?: string) => {
      const query = reason ? `?reason=${encodeURIComponent(reason)}` : "";
      return intent.run(
        () => apiFetch(`/segments/${interviewId}/${segmentId}${query}`, { method: "DELETE" }),
        (data) => !findLine(data, (line) => line.segment?.segment_id === segmentId),
      );
    },
    [intent, interviewId],
  );

  return { ...intent, removeSegment };
}

/** Flow 4: lens-item override. POST /lenses/{interview_id}/items/{item_id}/override */
export function useLensItemOverrideIntent(interviewId: string, transcriptQueryKey: QueryKey) {
  const intent = useCorrectionIntent({ queryKey: transcriptQueryKey });

  const overrideLensItem = useCallback(
    (itemId: string, fieldsOverridden: Record<string, unknown>, note?: string) =>
      intent.run(
        () =>
          apiFetch(`/lenses/${interviewId}/items/${itemId}/override`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ fields_overridden: fieldsOverridden, ...(note ? { note } : {}) }),
          }),
        (data) => {
          const transcript = data as TranscriptShape | undefined;
          return Boolean(
            transcript?.lines?.some((line) =>
              line.lens_items.some((item) => item.item_id === itemId && item.human_locked),
            ),
          );
        },
      ),
    [intent, interviewId],
  );

  return { ...intent, overrideLensItem };
}
