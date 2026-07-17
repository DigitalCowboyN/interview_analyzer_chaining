import type { TranscriptLineData } from "@/hooks/useTranscript";
import { useSentenceHistory } from "@/hooks/useSentenceHistory";
import { StateGate } from "@/components/StateGate";

export interface LineDetailPanelProps {
  interviewId: string;
  line: TranscriptLineData;
  onClose: () => void;
}

/**
 * Detail panel opened from a transcript line: full text, entities, lens
 * items (lens, node type, text, confidence, lock state), and edit history
 * fetched lazily (only while this panel is mounted/open — `useSentenceHistory`
 * is only called from here, so the request fires on open, not on every
 * transcript render).
 */
export function LineDetailPanel({
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
