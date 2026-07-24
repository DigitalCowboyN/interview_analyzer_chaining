import Link from "next/link";
import { queryKeys } from "@/hooks/queryKeys";
import {
  useEntityMergeAcceptIntent,
  useWorklistPersonLinkAcceptIntent,
  type CorrectionNotice,
  type FlowIntentPollOptions,
} from "@/hooks/mutations";
import type {
  WorklistData,
  WorklistLensItem,
  WorklistClaim,
  WorklistEntityMergeSuggestion,
  WorklistPersonLinkSuggestion,
} from "@/hooks/useWorklist";

export interface WorklistRowsProps {
  projectId: string;
  data: WorklistData;
  /** Poll-timing override for the accept intents (test injection point —
   * mirrors PersonPicker's pollOptions prop; defaults to production timing). */
  pollOptions?: FlowIntentPollOptions;
}

const DEGRADED_MESSAGE = "suggestions degraded — embedding provider unavailable";

function workbenchHref(projectId: string, interviewId: string): string {
  return `/workbench/${encodeURIComponent(projectId)}/${encodeURIComponent(interviewId)}`;
}

function NoticeText({ notice }: { notice: CorrectionNotice | null }) {
  if (!notice) return null;
  const tone = notice.kind === "timeout" ? "text-neutral-500" : "text-red-600";
  return (
    <p role={notice.kind === "timeout" ? "status" : "alert"} className={`mt-1 text-xs ${tone}`}>
      {notice.message}
    </p>
  );
}

function DegradationBanner({ flags }: { flags: string[] }) {
  if (!flags.includes("embedding_unavailable")) return null;
  return (
    <div
      role="status"
      className="mb-4 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-800"
    >
      {DEGRADED_MESSAGE}
    </div>
  );
}

function LensItemsSection({
  projectId,
  items,
}: {
  projectId: string;
  items: WorklistLensItem[];
}) {
  if (items.length === 0) return null;
  return (
    <section className="mt-6">
      <h2 className="text-xs font-semibold uppercase text-neutral-500">Lens items</h2>
      <ul className="mt-2 space-y-2">
        {items.map((item) => (
          <li key={item.item_id} className="rounded border border-neutral-200 p-3 text-sm">
            <Link
              href={workbenchHref(projectId, item.interview_id)}
              className="font-medium text-blue-700 hover:underline"
            >
              {item.lens} · {item.node_type}
            </Link>
            <div className="mt-1 text-xs text-neutral-500">
              {(item.confidence * 100).toFixed(0)}% confidence ·{" "}
              {item.reason.replace(/_/g, " ")}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ClaimsSection({
  projectId,
  claims,
}: {
  projectId: string;
  claims: WorklistClaim[];
}) {
  if (claims.length === 0) return null;
  return (
    <section className="mt-6">
      <h2 className="text-xs font-semibold uppercase text-neutral-500">Claims</h2>
      <ul className="mt-2 space-y-2">
        {claims.map((claim) => (
          <li key={claim.claim_id} className="rounded border border-neutral-200 p-3 text-sm">
            <Link
              href={workbenchHref(projectId, claim.interview_id)}
              className="font-medium text-blue-700 hover:underline"
            >
              {claim.text}
            </Link>
            <div className="mt-1 text-xs text-neutral-500">
              {claim.kind} · {(claim.confidence * 100).toFixed(0)}% confidence ·{" "}
              {claim.reason.replace(/_/g, " ")}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function EntityMergeSuggestionRow({
  projectId,
  suggestion,
  pollOptions,
}: {
  projectId: string;
  suggestion: WorklistEntityMergeSuggestion;
  pollOptions?: FlowIntentPollOptions;
}) {
  const { acceptMerge, isPending, status, notice } = useEntityMergeAcceptIntent(
    projectId,
    queryKeys.worklist(projectId),
    pollOptions,
  );

  if (status === "settled") return null;

  return (
    <li className="rounded border border-neutral-200 p-3 text-sm">
      <div className="flex items-center justify-between gap-2">
        <div>
          <span className="font-medium text-neutral-900">
            {suggestion.surfaces_a.join(", ")}
          </span>
          <span className="mx-1 text-neutral-400">↔</span>
          <span className="font-medium text-neutral-900">
            {suggestion.surfaces_b.join(", ")}
          </span>
        </div>
        <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs text-neutral-600">
          {suggestion.band}
        </span>
      </div>
      <div className="mt-1 text-xs text-neutral-500">score {suggestion.score.toFixed(2)}</div>
      <button
        type="button"
        disabled={isPending}
        onClick={() =>
          acceptMerge(suggestion.surviving_canonical_id, suggestion.merged_canonical_id)
        }
        className="mt-2 rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
      >
        {isPending ? "Accepting…" : "Accept merge"}
      </button>
      <NoticeText notice={notice} />
    </li>
  );
}

function EntityMergeSuggestionsSection({
  projectId,
  suggestions,
  pollOptions,
}: {
  projectId: string;
  suggestions: WorklistEntityMergeSuggestion[];
  pollOptions?: FlowIntentPollOptions;
}) {
  if (suggestions.length === 0) return null;
  return (
    <section className="mt-6">
      <h2 className="text-xs font-semibold uppercase text-neutral-500">
        Entity merge suggestions
      </h2>
      <ul className="mt-2 space-y-2">
        {suggestions.map((suggestion) => (
          <EntityMergeSuggestionRow
            key={`${suggestion.surviving_canonical_id}-${suggestion.merged_canonical_id}`}
            projectId={projectId}
            suggestion={suggestion}
            pollOptions={pollOptions}
          />
        ))}
      </ul>
    </section>
  );
}

function PersonLinkSuggestionRow({
  projectId,
  suggestion,
  pollOptions,
}: {
  projectId: string;
  suggestion: WorklistPersonLinkSuggestion;
  pollOptions?: FlowIntentPollOptions;
}) {
  const { acceptLink, isPending, status, notice } = useWorklistPersonLinkAcceptIntent(
    projectId,
    queryKeys.worklist(projectId),
    pollOptions,
  );

  if (status === "settled") return null;

  return (
    <li className="rounded border border-neutral-200 p-3 text-sm">
      <div className="font-medium text-neutral-900">{suggestion.display_name}</div>
      <div className="mt-1 text-xs text-neutral-500">
        {suggestion.speaker_display_name} · {suggestion.interview_id} · {suggestion.reason}
      </div>
      <button
        type="button"
        disabled={isPending}
        onClick={() =>
          acceptLink(
            suggestion.person_id,
            suggestion.interview_id,
            suggestion.speaker_id,
            suggestion.display_name,
          )
        }
        className="mt-2 rounded bg-neutral-900 px-2 py-1 text-xs text-white disabled:opacity-50"
      >
        {isPending ? "Linking…" : "Accept link"}
      </button>
      <NoticeText notice={notice} />
    </li>
  );
}

function PersonLinkSuggestionsSection({
  projectId,
  suggestions,
  pollOptions,
}: {
  projectId: string;
  suggestions: WorklistPersonLinkSuggestion[];
  pollOptions?: FlowIntentPollOptions;
}) {
  if (suggestions.length === 0) return null;
  return (
    <section className="mt-6">
      <h2 className="text-xs font-semibold uppercase text-neutral-500">
        Person link suggestions
      </h2>
      <ul className="mt-2 space-y-2">
        {suggestions.map((suggestion) => (
          <PersonLinkSuggestionRow
            key={`${suggestion.person_id}-${suggestion.interview_id}-${suggestion.speaker_id}`}
            projectId={projectId}
            suggestion={suggestion}
            pollOptions={pollOptions}
          />
        ))}
      </ul>
    </section>
  );
}

/**
 * The worklist's review rows (M5.0 Task 8): low-confidence lens items and
 * claims (link into the workbench transcript for manual review), plus
 * entity-merge and person-link suggestions with one-click accept affordances
 * built on the Task 5 intent pattern. Each accept row hides itself once its
 * own intent settles (the confirm-refetch has already confirmed the
 * suggestion is gone); the intent's invalidation of the worklist query key
 * keeps the shared cache consistent for the next full fetch.
 */
export function WorklistRows({ projectId, data, pollOptions }: WorklistRowsProps) {
  return (
    <div>
      <DegradationBanner flags={data.flags} />
      <LensItemsSection projectId={projectId} items={data.lens_items} />
      <ClaimsSection projectId={projectId} claims={data.claims} />
      <EntityMergeSuggestionsSection
        projectId={projectId}
        suggestions={data.entity_merge_suggestions}
        pollOptions={pollOptions}
      />
      <PersonLinkSuggestionsSection
        projectId={projectId}
        suggestions={data.person_link_suggestions}
        pollOptions={pollOptions}
      />
    </div>
  );
}
