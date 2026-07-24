import type { ReactNode } from "react";

interface StateGateProps {
  isLoading: boolean;
  isError: boolean;
  /** True when the loaded data is empty (e.g. an empty list) — checked after loading/error. */
  isEmpty?: boolean;
  error?: unknown;
  loadingFallback?: ReactNode;
  emptyFallback?: ReactNode;
  errorFallback?: ReactNode;
  children: ReactNode;
}

function defaultErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}

/**
 * The one shared loading/empty/error primitive. Every screen that fetches
 * data renders through this instead of hand-rolling its own state branching
 * — this is the error-handling doctrine: one place, one look and feel.
 *
 * Usage:
 *   <StateGate isLoading={isLoading} isError={isError} error={error} isEmpty={data?.length === 0}>
 *     {render actual content}
 *   </StateGate>
 */
export function StateGate({
  isLoading,
  isError,
  isEmpty = false,
  error,
  loadingFallback,
  emptyFallback,
  errorFallback,
  children,
}: StateGateProps) {
  if (isLoading) {
    return (
      <>{loadingFallback ?? <div role="status" className="p-4 text-sm text-neutral-500">Loading…</div>}</>
    );
  }

  if (isError) {
    return (
      <>
        {errorFallback ?? (
          <div role="alert" className="p-4 text-sm text-red-600">
            {defaultErrorMessage(error)}
          </div>
        )}
      </>
    );
  }

  if (isEmpty) {
    return (
      <>{emptyFallback ?? <div className="p-4 text-sm text-neutral-500">Nothing here yet.</div>}</>
    );
  }

  return <>{children}</>;
}
