import type { TranscriptMetadata } from "@/hooks/useTranscript";

/** Interview title strip. Front-matter metadata shows only when the graph has it. */
export function InterviewHeader({
  title,
  participants,
  date,
  lineCount,
  metadata,
}: {
  title: string;
  participants: string[];
  date: string;
  lineCount: number;
  metadata: TranscriptMetadata;
}) {
  const facts = [
    participants.join(", "),
    date,
    `${lineCount} ${lineCount === 1 ? "line" : "lines"}`,
  ].filter(Boolean);
  const entries = Object.entries(metadata);

  return (
    <div>
      <h1 className="text-xl font-semibold text-fg">{title}</h1>
      <p className="mt-1 text-sm text-fg-muted">{facts.join(" · ")}</p>
      {entries.length > 0 && (
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-fg-muted">{key}</dt>
              <dd className="text-fg">{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
