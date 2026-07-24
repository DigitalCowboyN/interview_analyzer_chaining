import { useState } from "react";
import type { TranscriptLineData } from "@/hooks/useTranscript";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { StateGate } from "@/components/StateGate";
import { PersonPicker } from "@/components/PersonPicker";
import { queryKeys } from "@/hooks/queryKeys";
import {
  useTextEditIntent,
  useSpeakerRenameIntent,
  useFragmentReattributeIntent,
  useSegmentRemoveIntent,
  useLensItemOverrideIntent,
  usePersonUnlinkIntent,
  type CorrectionNotice,
} from "@/hooks/mutations";

export interface LineDetailPanelProps {
  projectId: string;
  interviewId: string;
  line: TranscriptLineData;
  onClose: () => void;
}

/** Inline notice for a correction's terminal non-settled state (timeout/conflict/network). */
function CorrectionNoticeBanner({ notice }: { notice: CorrectionNotice | null }) {
  if (!notice) return null;
  const tone = notice.kind === "timeout" ? "text-neutral-500" : "text-red-600";
  return (
    <p role={notice.kind === "timeout" ? "status" : "alert"} className={`mt-1 text-xs ${tone}`}>
      {notice.message}
    </p>
  );
}

/** Flow 1: edit-in-place for the line's text. */
function TextEditControl({
  interviewId,
  line,
}: {
  interviewId: string;
  line: TranscriptLineData;
}) {
  const { editText, isPending, notice } = useTextEditIntent(
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(line.text);

  if (!editing) {
    return (
      <div className="mt-2">
        <button
          type="button"
          onClick={() => {
            setDraft(line.text);
            setEditing(true);
          }}
          className="text-xs text-blue-700 hover:underline"
        >
          Edit text
        </button>
        <CorrectionNoticeBanner notice={notice} />
      </div>
    );
  }

  return (
    <div className="mt-2">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        className="w-full rounded border border-neutral-300 p-2 text-sm"
        rows={3}
        aria-label="Edit sentence text"
      />
      <div className="mt-1 flex gap-2">
        <button
          type="button"
          disabled={isPending || draft.trim().length === 0}
          onClick={async () => {
            const outcome = await editText(line.sequence_order, draft);
            if (outcome.status === "settled") setEditing(false);
          }}
          className="rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {isPending ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => setEditing(false)}
          className="rounded border border-neutral-300 px-2 py-1 text-xs"
        >
          Cancel
        </button>
      </div>
      <CorrectionNoticeBanner notice={notice} />
    </div>
  );
}

/** Flow 2a: speaker rename. */
function SpeakerRenameControl({
  interviewId,
  line,
}: {
  interviewId: string;
  line: TranscriptLineData;
}) {
  const { renameSpeaker, isPending, notice } = useSpeakerRenameIntent(
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(line.speaker?.display_name ?? "");

  if (!line.speaker) return null;

  if (!editing) {
    return (
      <div className="mt-2">
        <button
          type="button"
          onClick={() => {
            setDraft(line.speaker!.display_name);
            setEditing(true);
          }}
          className="text-xs text-blue-700 hover:underline"
        >
          Rename speaker
        </button>
        <CorrectionNoticeBanner notice={notice} />
      </div>
    );
  }

  return (
    <div className="mt-2">
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        className="w-full rounded border border-neutral-300 p-1 text-sm"
        aria-label="New speaker display name"
      />
      <div className="mt-1 flex gap-2">
        <button
          type="button"
          disabled={isPending || draft.trim().length === 0}
          onClick={async () => {
            const outcome = await renameSpeaker(line.speaker!.speaker_id, draft.trim());
            if (outcome.status === "settled") setEditing(false);
          }}
          className="rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {isPending ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => setEditing(false)}
          className="rounded border border-neutral-300 px-2 py-1 text-xs"
        >
          Cancel
        </button>
      </div>
      <CorrectionNoticeBanner notice={notice} />
    </div>
  );
}

/**
 * Flow 2b: reattribute this fragment to a different (existing) speaker id.
 * A proper speaker picker is Task 6's concern (the manual person-linking
 * picker) — this is the minimal raw affordance: enter the target speaker id.
 */
function FragmentReattributeControl({
  interviewId,
  line,
}: {
  interviewId: string;
  line: TranscriptLineData;
}) {
  const { reattributeFragment, isPending, notice } = useFragmentReattributeIntent(
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  if (!editing) {
    return (
      <div className="mt-2">
        <button
          type="button"
          onClick={() => {
            setDraft("");
            setEditing(true);
          }}
          className="text-xs text-blue-700 hover:underline"
        >
          Reattribute to another speaker
        </button>
        <CorrectionNoticeBanner notice={notice} />
      </div>
    );
  }

  return (
    <div className="mt-2">
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="Target speaker id"
        className="w-full rounded border border-neutral-300 p-1 text-sm"
        aria-label="Target speaker id"
      />
      <div className="mt-1 flex gap-2">
        <button
          type="button"
          disabled={isPending || draft.trim().length === 0}
          onClick={async () => {
            const outcome = await reattributeFragment(line.sequence_order, draft.trim());
            if (outcome.status === "settled") setEditing(false);
          }}
          className="rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {isPending ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => setEditing(false)}
          className="rounded border border-neutral-300 px-2 py-1 text-xs"
        >
          Cancel
        </button>
      </div>
      <CorrectionNoticeBanner notice={notice} />
    </div>
  );
}

/**
 * Flow 6 (M5.0 Task 6): manual speaker→person linking — the identity escape
 * hatch for speakers automation can't reach. Unlinked speakers get an
 * "identify as person…" affordance opening the PersonPicker; linked
 * speakers (person suffix already shown per Task 4) get an unlink
 * affordance instead.
 */
function PersonLinkControl({
  projectId,
  interviewId,
  line,
}: {
  projectId: string;
  interviewId: string;
  line: TranscriptLineData;
}) {
  const { unlinkPerson, isPending, notice } = usePersonUnlinkIntent(
    projectId,
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [picking, setPicking] = useState(false);

  if (!line.speaker) return null;

  if (line.person) {
    return (
      <div className="mt-2">
        <button
          type="button"
          disabled={isPending}
          onClick={() => unlinkPerson(line.person!.person_id, line.speaker!.speaker_id)}
          className="text-xs text-red-700 hover:underline disabled:opacity-50"
        >
          {isPending ? "Unlinking…" : "Unlink person"}
        </button>
        <PersonNoticeBanner notice={notice} />
      </div>
    );
  }

  if (picking) {
    return (
      <PersonPicker
        projectId={projectId}
        interviewId={interviewId}
        speakerId={line.speaker.speaker_id}
        onClose={() => setPicking(false)}
        onLinked={() => setPicking(false)}
      />
    );
  }

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setPicking(true)}
        className="text-xs text-blue-700 hover:underline"
      >
        Identify as person…
      </button>
    </div>
  );
}

function PersonNoticeBanner({ notice }: { notice: CorrectionNotice | null }) {
  if (!notice) return null;
  const tone = notice.kind === "timeout" ? "text-neutral-500" : "text-red-600";
  return (
    <p role={notice.kind === "timeout" ? "status" : "alert"} className={`mt-1 text-xs ${tone}`}>
      {notice.message}
    </p>
  );
}

/** Flow 3: segment remove (only rendered when the line belongs to a segment). */
function SegmentRemoveControl({
  interviewId,
  line,
}: {
  interviewId: string;
  line: TranscriptLineData;
}) {
  const { removeSegment, isPending, notice } = useSegmentRemoveIntent(
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [confirming, setConfirming] = useState(false);
  const [reason, setReason] = useState("");

  if (!line.segment) return null;

  if (!confirming) {
    return (
      <div className="mt-2">
        <button
          type="button"
          onClick={() => setConfirming(true)}
          className="text-xs text-red-700 hover:underline"
        >
          Remove segment
        </button>
        <CorrectionNoticeBanner notice={notice} />
      </div>
    );
  }

  return (
    <div className="mt-2">
      <input
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="Reason (optional)"
        className="w-full rounded border border-neutral-300 p-1 text-sm"
        aria-label="Reason for removing this segment"
      />
      <div className="mt-1 flex gap-2">
        <button
          type="button"
          disabled={isPending}
          onClick={async () => {
            const outcome = await removeSegment(
              line.segment!.segment_id,
              reason.trim() || undefined,
            );
            if (outcome.status === "settled") setConfirming(false);
          }}
          className="rounded bg-red-700 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {isPending ? "Removing…" : "Confirm remove"}
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => setConfirming(false)}
          className="rounded border border-neutral-300 px-2 py-1 text-xs"
        >
          Cancel
        </button>
      </div>
      <CorrectionNoticeBanner notice={notice} />
    </div>
  );
}

/** Flow 4: lens-item override — one control per lens item. */
function LensItemOverrideControl({
  interviewId,
  itemId,
  currentText,
}: {
  interviewId: string;
  itemId: string;
  currentText: string;
}) {
  const { overrideLensItem, isPending, notice } = useLensItemOverrideIntent(
    interviewId,
    queryKeys.transcript(interviewId),
  );
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(currentText);

  if (!editing) {
    return (
      <div className="mt-1">
        <button
          type="button"
          onClick={() => {
            setDraft(currentText);
            setEditing(true);
          }}
          className="text-xs text-blue-700 hover:underline"
        >
          Correct
        </button>
        <CorrectionNoticeBanner notice={notice} />
      </div>
    );
  }

  return (
    <div className="mt-1">
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        className="w-full rounded border border-neutral-300 p-1 text-sm"
        aria-label="Corrected lens item text"
      />
      <div className="mt-1 flex gap-2">
        <button
          type="button"
          disabled={isPending || draft.trim().length === 0}
          onClick={async () => {
            const outcome = await overrideLensItem(itemId, { text: draft.trim() });
            if (outcome.status === "settled") setEditing(false);
          }}
          className="rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {isPending ? "Saving…" : "Save"}
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() => setEditing(false)}
          className="rounded border border-neutral-300 px-2 py-1 text-xs"
        >
          Cancel
        </button>
      </div>
      <CorrectionNoticeBanner notice={notice} />
    </div>
  );
}

/**
 * Detail panel opened from a transcript line: full text, entities, lens
 * items (lens, node type, text, confidence, lock state), edit history
 * fetched lazily, the four correction affordances (M5.0 Task 5): text edit,
 * speaker rename, segment remove, lens-item override — plus manual
 * speaker→person linking (M5.0 Task 6): identify/unlink. Each
 * hooks in `@/hooks/mutations` — this component stays prop-driven for its
 * own line data; all write state lives in those hooks.
 */
export function LineDetailPanel({
  projectId,
  interviewId,
  line,
  onClose,
}: LineDetailPanelProps) {
  const {
    data: history,
    isLoading,
    isError,
    error,
  } = useSentenceHistory(interviewId, line.sequence_order, true);

  return (
    <aside
      role="dialog"
      aria-label="Line detail"
      className="w-96 shrink-0 border-l border-neutral-200 p-4"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-neutral-900">Line detail</h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close detail panel"
          className="text-neutral-500 hover:text-neutral-900"
        >
          ×
        </button>
      </div>

      <p className="mt-3 text-sm text-neutral-900">{line.text}</p>
      <TextEditControl interviewId={interviewId} line={line} />

      <section className="mt-4">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Speaker
        </h3>
        <SpeakerRenameControl interviewId={interviewId} line={line} />
        {line.speaker && <FragmentReattributeControl interviewId={interviewId} line={line} />}
        <PersonLinkControl projectId={projectId} interviewId={interviewId} line={line} />
      </section>

      <section className="mt-4">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Segment
        </h3>
        {line.segment ? (
          <>
            <p className="mt-1 text-sm text-neutral-900">
              {line.segment.topic ?? "Untitled segment"}
            </p>
            <SegmentRemoveControl interviewId={interviewId} line={line} />
          </>
        ) : (
          <p className="mt-1 text-sm text-neutral-400">Not part of a segment.</p>
        )}
      </section>

      <section className="mt-4">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Entities
        </h3>
        {line.entities.length === 0 ? (
          <p className="mt-1 text-sm text-neutral-400">No entities.</p>
        ) : (
          <ul className="mt-1 space-y-1 text-sm">
            {line.entities.map((entity, index) => (
              <li key={`${entity.surface}-${index}`}>
                {entity.surface}{" "}
                <span className="text-neutral-500">({entity.entity_type})</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-4">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Lens items
        </h3>
        {line.lens_items.length === 0 ? (
          <p className="mt-1 text-sm text-neutral-400">No lens items.</p>
        ) : (
          <ul className="mt-1 space-y-2 text-sm">
            {line.lens_items.map((item) => (
              <li key={item.item_id} className="rounded border border-neutral-200 p-2">
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  <span className="font-medium text-neutral-700">{item.lens}</span>
                  <span>{item.node_type}</span>
                  <span>{(item.confidence * 100).toFixed(0)}%</span>
                  {item.human_locked && (
                    <span className="rounded bg-neutral-200 px-1.5 py-0.5">locked</span>
                  )}
                </div>
                <p className="mt-1 text-neutral-900">{item.text}</p>
                <LensItemOverrideControl
                  interviewId={interviewId}
                  itemId={item.item_id}
                  currentText={item.text}
                />
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-4">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Edit history
        </h3>
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={history?.events.length === 0}
          emptyFallback={<p className="mt-1 text-sm text-neutral-400">No edits yet.</p>}
        >
          <ul className="mt-1 space-y-1 text-sm">
            {history?.events.map((event, index) => (
              <li key={`${event.correlation_id}-${index}`} className="text-neutral-700">
                <span className="font-medium">{event.event_type}</span>{" "}
                <span className="text-neutral-500">
                  v{event.version} · {event.actor.user_id} · {event.occurred_at}
                </span>
              </li>
            ))}
          </ul>
        </StateGate>
      </section>
    </aside>
  );
}
