export interface NoticeTextProps {
  notice: { kind: string; message: string } | null;
  /** Vertical margin utility for the call site (the four near-identical
   * copies this replaces varied only between `mt-1` and `mt-2`). */
  className?: string;
}

/**
 * Shared inline notice for a correction intent's terminal non-settled state
 * (timeout/conflict/network) — the ONE renderer for what were four
 * byte-/near-identical copies (LineDetailPanel's CorrectionNoticeBanner and
 * PersonNoticeBanner, PersonPicker's PersonPickerNotice, WorklistRows'
 * NoticeText). `role="status"` for the non-error "timeout" tone (still
 * processing, not a failure); `role="alert"` for conflict/network.
 */
export function NoticeText({ notice, className = "mt-1" }: NoticeTextProps) {
  if (!notice) return null;
  const tone = notice.kind === "timeout" ? "text-neutral-500" : "text-red-600";
  return (
    <p role={notice.kind === "timeout" ? "status" : "alert"} className={`${className} text-xs ${tone}`}>
      {notice.message}
    </p>
  );
}
