import type { LiveStatus } from "@/hooks/useLiveInvalidation";

export interface LiveIndicatorProps {
  status: LiveStatus;
}

const LABEL: Record<LiveStatus, string> = {
  live: "Live updates on",
  offline: "Live updates off",
  idle: "Live updates off",
};

/**
 * Subtle presentational dot + accessible label for the live-invalidation
 * connection state (M5.1 Task 5). `aria-live="polite"` makes status flips
 * (live <-> offline) announced to assistive tech without stealing focus;
 * deliberately NOT `role="status"` — that role is already used by
 * StateGate's loading indicator on the same page, and a second one would
 * make `getByRole("status")` ambiguous for every existing page test. The
 * dot itself is decorative (`aria-hidden`) since the text carries the
 * meaning. No hooks beyond props — components never touch EventSource
 * directly, only `useLiveInvalidation`'s returned status.
 */
export function LiveIndicator({ status }: LiveIndicatorProps) {
  const isLive = status === "live";
  return (
    <span aria-live="polite" className="inline-flex items-center gap-1.5 text-xs text-neutral-500">
      <span
        aria-hidden="true"
        className={`h-2 w-2 rounded-full ${isLive ? "bg-emerald-500" : "bg-neutral-300"}`}
      />
      <span>{LABEL[status]}</span>
    </span>
  );
}
