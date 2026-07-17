import { useState } from "react";
import { usePersons, usePersonId } from "@/hooks/usePersons";
import { usePersonLinkIntent, type CorrectionNotice, type FlowIntentPollOptions } from "@/hooks/mutations";
import { queryKeys } from "@/hooks/queryKeys";
import { StateGate } from "@/components/StateGate";

export interface PersonPickerProps {
  projectId: string;
  interviewId: string;
  speakerId: string;
  onClose: () => void;
  /** Called once the link settles — lets the caller close the picker. */
  onLinked?: () => void;
  /** Poll-timing override for the underlying intent (test injection point —
   * mirrors the Task 5 flow-hook pattern; defaults to production timing). */
  pollOptions?: FlowIntentPollOptions;
}

/**
 * "Identify as person…" picker (M5.0 Task 6): lists the project's existing
 * persons for one-click linking, plus a create-new path (name → server-side
 * derivation → link with display_name so the backend mints the person).
 * The frontend NEVER derives person ids itself (loose-coupling requirement).
 */
export function PersonPicker({
  projectId,
  interviewId,
  speakerId,
  onClose,
  onLinked,
  pollOptions,
}: PersonPickerProps) {
  const { data: persons, isLoading, isError, error } = usePersons(projectId);
  const { derivePersonId } = usePersonId();
  const { linkPerson, isPending, notice } = usePersonLinkIntent(
    projectId,
    interviewId,
    queryKeys.transcript(interviewId),
    pollOptions,
  );
  const [newName, setNewName] = useState("");
  const [deriving, setDeriving] = useState(false);

  async function handleLinkExisting(personId: string) {
    const outcome = await linkPerson(personId, speakerId);
    if (outcome.status === "settled") onLinked?.();
  }

  async function handleCreateNew() {
    const displayName = newName.trim();
    if (!displayName) return;
    setDeriving(true);
    try {
      const personId = await derivePersonId(projectId, displayName);
      const outcome = await linkPerson(personId, speakerId, displayName);
      if (outcome.status === "settled") {
        setNewName("");
        onLinked?.();
      }
    } finally {
      setDeriving(false);
    }
  }

  const busy = isPending || deriving;

  return (
    <div role="dialog" aria-label="Identify as person" className="mt-2 rounded border border-neutral-300 p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-neutral-500">
          Identify as person
        </h3>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close person picker"
          className="text-neutral-500 hover:text-neutral-900"
        >
          ×
        </button>
      </div>

      <StateGate
        isLoading={isLoading}
        isError={isError}
        error={error}
        isEmpty={persons?.length === 0}
        emptyFallback={<p className="mt-2 text-sm text-neutral-400">No persons yet.</p>}
      >
        <ul className="mt-2 space-y-1">
          {persons?.map((person) => (
            <li key={person.person_id}>
              <button
                type="button"
                disabled={busy}
                onClick={() => handleLinkExisting(person.person_id)}
                className="w-full rounded border border-neutral-200 px-2 py-1 text-left text-sm hover:bg-neutral-50 disabled:opacity-50"
              >
                {person.display_name}
              </button>
            </li>
          ))}
        </ul>
      </StateGate>

      <div className="mt-3 border-t border-neutral-200 pt-3">
        <label className="text-xs font-semibold uppercase text-neutral-500" htmlFor="new-person-name">
          Create new person
        </label>
        <input
          id="new-person-name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="Display name"
          className="mt-1 w-full rounded border border-neutral-300 p-1 text-sm"
          aria-label="New person display name"
        />
        <button
          type="button"
          disabled={busy || newName.trim().length === 0}
          onClick={handleCreateNew}
          className="mt-1 rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
        >
          {busy ? "Linking…" : "Create and link"}
        </button>
      </div>

      <PersonPickerNotice notice={notice} />
    </div>
  );
}

function PersonPickerNotice({ notice }: { notice: CorrectionNotice | null }) {
  if (!notice) return null;
  const tone = notice.kind === "timeout" ? "text-neutral-500" : "text-red-600";
  return (
    <p role={notice.kind === "timeout" ? "status" : "alert"} className={`mt-2 text-xs ${tone}`}>
      {notice.message}
    </p>
  );
}
