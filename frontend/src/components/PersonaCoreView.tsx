import type { PersonaDimensionItem, PersonaDimensions } from "@/hooks/usePersonaDetail";

const DIMENSION_LABELS: { key: keyof PersonaDimensions; label: string }[] = [
  { key: "traits", label: "Traits" },
  { key: "goals", label: "Goals" },
  { key: "pain_points", label: "Pain points" },
  { key: "notable_quotes", label: "Notable quotes" },
];

function DimensionItemRow({ item }: { item: PersonaDimensionItem }) {
  return (
    <li className="flex flex-wrap items-center justify-between gap-2 rounded border border-border p-3 bg-surface">
      <span className="text-sm text-fg">{item.text}</span>
      <span className="flex items-center gap-2">
        <span
          title={`Confidence ${item.confidence}`}
          className="rounded bg-surface-raised px-1.5 py-0.5 text-xs text-fg-muted"
        >
          {Math.round(item.confidence * 100)}%
        </span>
        <span className="rounded bg-accent-subtle px-1.5 py-0.5 text-xs text-accent">
          {item.interview_title}
        </span>
      </span>
    </li>
  );
}

/**
 * Persona CORE view (not a card): dimension-grouped items with per-interview
 * provenance chips. Distinct route `projects/[projectId]/personas/[personId]`
 * — Persona is its own entity type, never embedded in the person core view.
 */
export function PersonaCoreView({ dimensions }: { dimensions: PersonaDimensions }) {
  return (
    <div className="space-y-6">
      {DIMENSION_LABELS.map(({ key, label }) => {
        const items = dimensions[key];
        return (
          <section key={key}>
            <h2 className="text-sm font-semibold uppercase text-fg-muted">
              {label} ({items.length})
            </h2>
            {items.length === 0 ? (
              <p className="mt-2 text-sm text-fg-muted">None recorded.</p>
            ) : (
              <ul className="mt-2 space-y-2">
                {items.map((item) => (
                  <DimensionItemRow key={item.item_id} item={item} />
                ))}
              </ul>
            )}
          </section>
        );
      })}
    </div>
  );
}
