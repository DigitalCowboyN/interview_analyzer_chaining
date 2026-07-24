import type { TranscriptMetadata } from "@/hooks/useTranscript";

/**
 * Title + front-matter fields. KNOWN BACKEND GAP: `metadata` is currently
 * always `{}` (front matter isn't projected onto the graph yet — see
 * src/ui/reader.py::interview_header_row), so the common case is the quiet
 * empty state below rather than an error.
 */
export function MetadataPanel({
  title,
  metadata,
}: {
  title: string;
  metadata: TranscriptMetadata;
}) {
  const entries = Object.entries(metadata);

  return (
    <div className="rounded border border-neutral-200 p-4">
      <h1 className="text-lg font-semibold text-neutral-900">{title}</h1>
      {entries.length === 0 ? (
        <p className="mt-2 text-sm text-neutral-400">
          No metadata available.
        </p>
      ) : (
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-neutral-500">{key}</dt>
              <dd className="text-neutral-900">{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
