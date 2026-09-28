import { groupInsights, type Insight } from "@/lib/insights";

/** Interview-level lens output (spec §B), grouped by type in canonical order. */
export function InsightsPanel({
  insights,
  selectedId,
  onSelect,
  isLoading,
}: {
  insights: Insight[];
  selectedId: string | null;
  onSelect: (insight: Insight) => void;
  isLoading?: boolean;
}) {
  if (isLoading) return <p className="p-4 text-sm text-fg-muted">Loading insights…</p>;

  const groups = groupInsights(insights);
  if (groups.length === 0) return <p className="p-4 text-sm text-fg-muted">No insights yet.</p>;

  return (
    <div className="space-y-5">
      {groups.map((group) => (
        <section key={group.nodeType}>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-fg-muted">
            {group.label} ({group.items.length})
          </h2>
          <ul className="mt-2 space-y-1">
            {group.items.map((item) => {
              const selected = item.item_id === selectedId;
              return (
                <li key={item.item_id}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => onSelect(item)}
                    className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${
                      selected ? "bg-accent-subtle text-fg" : "text-fg hover:bg-surface-raised"
                    }`}
                  >
                    {item.text}
                    <span className="ml-2 text-xs text-fg-muted">{Math.round(item.confidence * 100)}%</span>
                    {item.locked && <span className="ml-2 text-xs text-warning">locked</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
