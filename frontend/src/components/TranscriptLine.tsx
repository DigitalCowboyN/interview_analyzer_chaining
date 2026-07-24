import type { TranscriptLineData } from "@/hooks/useTranscript";

function speakerLabel(line: TranscriptLineData): string | null {
  if (!line.speaker) return null;
  // "Speaker (Person)" pattern — mirrors src/ask/context.py::build_blocks'
  // speaker_line convention when a person is linked.
  return line.person
    ? `${line.speaker.display_name} (${line.person.display_name})`
    : line.speaker.display_name;
}

export interface TranscriptLineProps {
  line: TranscriptLineData;
  /** True when this line shares its utterance_id with the previous rendered
   * line — the visual cue for utterance grouping (no top border/gap, so
   * grouped lines read as one continuous turn). */
  continuesUtterance: boolean;
  onSelect: (line: TranscriptLineData) => void;
}

/** One transcript line: speaker (+ person suffix), edited badge, click-to-open detail. */
export function TranscriptLine({
  line,
  continuesUtterance,
  onSelect,
}: TranscriptLineProps) {
  const label = speakerLabel(line);

  return (
    <button
      type="button"
      onClick={() => onSelect(line)}
      data-utterance-id={line.utterance_id ?? undefined}
      data-continues-utterance={continuesUtterance}
      className={`block w-full text-left px-3 py-2 hover:bg-neutral-50 ${
        continuesUtterance
          ? "border-l-2 border-neutral-300 ml-3"
          : "border-l-2 border-transparent mt-2"
      }`}
    >
      <div className="flex items-center gap-2 text-xs text-neutral-500">
        {label && <span className="font-medium text-neutral-700">{label}</span>}
        {line.edited && (
          <span className="rounded bg-amber-100 px-1.5 py-0.5 text-amber-800">
            edited
          </span>
        )}
      </div>
      <p className="text-sm text-neutral-900">{line.text}</p>
    </button>
  );
}
