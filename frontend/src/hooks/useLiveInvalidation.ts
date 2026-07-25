import { useEffect, useRef, useState } from "react";
import type { QueryClient, QueryKey } from "@tanstack/react-query";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "@/hooks/queryKeys";

/**
 * Frontend live-invalidation layer (M5.1 Task 5). The frontend knows ONLY
 * the thin SSE contract shipped in Task 4
 * (src/api/routers/ui.py::stream_events / src/ui/notifications.py): each
 * message is `{"surface": "transcript"|"interviews"|"project"|"resync",
 * interview_id?, project_id?}`. Nothing upstream of this module — no ESDB,
 * no projection names, no stream types.
 *
 * The backend only ever DELIVERS a notification whose id(s) match this
 * subscription's own scope exactly (NotificationHub._matches subscribes by
 * interview_id/project_id and filters server-side), so the mapping below
 * targets the HOOK's own scopes rather than re-deriving ids off the message
 * payload — the payload's ids, when present, are guaranteed equal to ours.
 */

export type LiveStatus = "live" | "offline" | "idle";

export interface LiveInvalidationScopes {
  interviewId?: string;
  projectId?: string;
  personId?: string;
}

export interface LiveInvalidationTiming {
  coalesceMs?: number;
  trailingMs?: number;
}

/** Production defaults — pinned by a test (M5.1 Task 5 binding spec). */
export const DEFAULT_COALESCE_MS = 500;
export const DEFAULT_TRAILING_MS = 2000;

interface StreamNotification {
  surface: string;
  interview_id?: string;
  project_id?: string;
}

/** Relative URL through the Next proxy — EventSource cannot set headers and
 * none are needed (see src/api/routers/ui.py::stream_events docstring). */
export function buildStreamUrl(scopes: LiveInvalidationScopes): string {
  const params = new URLSearchParams();
  if (scopes.interviewId) params.set("interview_id", scopes.interviewId);
  if (scopes.projectId) params.set("project_id", scopes.projectId);
  return `/api/ui/streams/events?${params.toString()}`;
}

/**
 * Notification -> affected query keys, per the binding mapping:
 *   transcript -> transcript(interviewId)
 *   interviews -> interviews(projectId)
 *   project    -> transcript(interviewId) IF an interviewId scope is present
 *                 (person links affect the open transcript), AND
 *                 persons(projectId), personas(projectId), worklist(projectId);
 *                 PLUS persona(projectId, personId) and person(projectId, personId)
 *                 IF a personId scope is present (M5.1b gallery liveness)
 *   resync     -> every key this hook watches (union of the above, given the
 *                 hook's current scopes)
 * Pure function — exported for direct unit testing.
 */
export function keysForSurface(surface: string, scopes: LiveInvalidationScopes): QueryKey[] {
  const keys: QueryKey[] = [];
  const { interviewId, projectId, personId } = scopes;

  switch (surface) {
    case "transcript":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      break;
    case "interviews":
      if (projectId) keys.push(queryKeys.interviews(projectId));
      break;
    case "project":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      if (projectId) {
        keys.push(queryKeys.persons(projectId));
        keys.push(queryKeys.personas(projectId));
        keys.push(queryKeys.worklist(projectId));
        if (personId) {
          keys.push(queryKeys.persona(projectId, personId));
          keys.push(queryKeys.person(projectId, personId));
        }
      }
      break;
    case "resync":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      if (projectId) {
        keys.push(queryKeys.interviews(projectId));
        keys.push(queryKeys.persons(projectId));
        keys.push(queryKeys.personas(projectId));
        keys.push(queryKeys.worklist(projectId));
        if (personId) {
          keys.push(queryKeys.persona(projectId, personId));
          keys.push(queryKeys.person(projectId, personId));
        }
      }
      break;
    default:
      break;
  }
  return keys;
}

interface KeyDebounceState {
  /** Non-null while this key is within its post-immediate coalesce window;
   * further notifications during this window are absorbed (no immediate
   * re-invalidate). */
  cooldown: ReturnType<typeof setTimeout> | null;
  /** Always reset (cleared + restarted) on every notification for this key —
   * the projection-race mitigation: a trailing re-invalidate fires
   * `trailingMs` after the MOST RECENT notification, doctrine not
   * optimization. */
  trailing: ReturnType<typeof setTimeout> | null;
}

/**
 * Per-query-key debounced dispatcher: first notification for a key
 * invalidates immediately; further notifications within `coalesceMs` are
 * coalesced; a trailing invalidate always fires `trailingMs` after the most
 * recent notification (timer resets per notification). Owns its own timer
 * map so callers just call `.notify(key)` per affected key and `.dispose()`
 * on cleanup.
 */
function createKeyDebouncer(
  queryClient: QueryClient,
  coalesceMs: number,
  trailingMs: number,
) {
  const states = new Map<string, KeyDebounceState>();

  const invalidate = (key: QueryKey) => {
    queryClient.invalidateQueries({ queryKey: key, exact: true });
  };

  return {
    notify(key: QueryKey) {
      const keyId = JSON.stringify(key);
      let state = states.get(keyId);
      if (!state) {
        state = { cooldown: null, trailing: null };
        states.set(keyId, state);
      }

      if (!state.cooldown) {
        invalidate(key);
        state.cooldown = setTimeout(() => {
          state!.cooldown = null;
        }, coalesceMs);
      }

      if (state.trailing) clearTimeout(state.trailing);
      state.trailing = setTimeout(() => {
        invalidate(key);
        state!.trailing = null;
      }, trailingMs);
    },
    dispose() {
      for (const state of states.values()) {
        if (state.cooldown) clearTimeout(state.cooldown);
        if (state.trailing) clearTimeout(state.trailing);
      }
      states.clear();
    },
  };
}

/**
 * One EventSource per mount, scoped to the given interview/project. No
 * scopes -> "idle", no connection. `onopen` -> "live", `onerror` ->
 * "offline" (the browser auto-retries natively; status flips back to "live"
 * on the next `onopen` — no custom reconnect logic here). Closed on unmount.
 */
export function useLiveInvalidation(
  scopes: LiveInvalidationScopes,
  timing?: LiveInvalidationTiming,
): LiveStatus {
  const { interviewId, projectId, personId } = scopes;
  const coalesceMs = timing?.coalesceMs ?? DEFAULT_COALESCE_MS;
  const trailingMs = timing?.trailingMs ?? DEFAULT_TRAILING_MS;
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<LiveStatus>("idle");
  const wasOfflineRef = useRef(false);

  useEffect(() => {
    if (!interviewId && !projectId) {
      setStatus("idle");
      return;
    }

    wasOfflineRef.current = false;
    const debouncer = createKeyDebouncer(queryClient, coalesceMs, trailingMs);
    const source = new EventSource(buildStreamUrl({ interviewId, projectId }));

    source.onopen = () => {
      setStatus("live");
      // Reconnect after a browser<->backend outage: notifications published
      // during the gap are gone (the backend resubscribes from_end, no
      // replay), so catch up by invalidating everything this hook watches —
      // otherwise the badge reads "live" over stale data. First connect
      // (never offline) skips this.
      if (wasOfflineRef.current) {
        wasOfflineRef.current = false;
        for (const key of keysForSurface("resync", { interviewId, projectId, personId })) {
          debouncer.notify(key);
        }
      }
    };
    source.onerror = () => {
      wasOfflineRef.current = true;
      setStatus("offline");
    };
    source.onmessage = (event: MessageEvent<string>) => {
      let parsed: StreamNotification;
      try {
        parsed = JSON.parse(event.data);
      } catch {
        return; // unparsable — ignore per contract
      }
      for (const key of keysForSurface(parsed.surface, { interviewId, projectId, personId })) {
        debouncer.notify(key);
      }
    };

    return () => {
      source.close();
      debouncer.dispose();
    };
  }, [interviewId, projectId, personId, coalesceMs, trailingMs, queryClient]);

  return status;
}
