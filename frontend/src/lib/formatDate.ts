const FORMAT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

/** "Sep 27, 2026". Trims >3 fractional-second digits (Neo4j emits 6), which
 * some engines refuse to parse; unparseable input comes back unchanged. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const normalized = iso.replace(/(\.\d{3})\d+/, "$1");
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? iso : FORMAT.format(date);
}
