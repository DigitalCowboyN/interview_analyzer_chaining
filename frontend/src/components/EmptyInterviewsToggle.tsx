interface WithFragmentCount {
  fragment_count: number;
}

/** Splits interview-like rows into those with transcript lines and a count of those without. */
export function splitEmpty<T extends WithFragmentCount>(
  items: T[],
): { withLines: T[]; emptyCount: number } {
  const withLines = items.filter((i) => i.fragment_count > 0);
  return { withLines, emptyCount: items.length - withLines.length };
}

interface EmptyInterviewsToggleProps {
  /** Count of rows with no transcript lines. */
  emptyCount: number;
  /** True when no row in the (unfiltered) list has any lines. */
  allEmpty: boolean;
  showEmpty: boolean;
  onToggle: () => void;
  /** Where these rows live, for the all-empty message. Defaults to "this project". */
  scope?: string;
}

/** Presentational: the "Show N empty" / "Hide empty interviews" toggle, plus
 * an explanatory message when every row in scope has no transcript lines. */
export function EmptyInterviewsToggle({
  emptyCount,
  allEmpty,
  showEmpty,
  onToggle,
  scope = "this project",
}: EmptyInterviewsToggleProps) {
  if (emptyCount === 0) return null;

  return (
    <div className="mt-3">
      {allEmpty && (
        <p className="mb-2 text-sm text-fg-muted">
          {emptyCount === 1
            ? `All 1 interview in ${scope} has no transcript lines yet.`
            : `All ${emptyCount} interviews in ${scope} have no transcript lines yet.`}
        </p>
      )}
      <button type="button" onClick={onToggle} className="text-sm text-accent">
        {showEmpty ? "Hide empty interviews" : `Show ${emptyCount} empty`}
      </button>
    </div>
  );
}
