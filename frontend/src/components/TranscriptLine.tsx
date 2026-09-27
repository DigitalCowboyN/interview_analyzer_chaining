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
  /** True when this line is a supporting fragment of the currently selected
   * insight — scrolled to and visually highlighted. */
  highlighted?: boolean;
}

/** One transcript line: speaker (+ person suffix), edited badge, click-to-open detail. */
export function TranscriptLine({
  line,
  continuesUtterance,
  onSelect,
  highlighted,
}: TranscriptLineProps) {
  const label = speakerLabel(line);

  return (
    <button
      type="button"
      id={`line-${line.fragment_id}`}
      onClick={() => onSelect(line)}
      data-utterance-id={line.utterance_id ?? undefined}
      data-continues-utterance={continuesUtterance}
      data-highlighted={highlighted ? "true" : undefined}
      className={`block w-full text-left px-3 py-2 hover:bg-surface-raised ${
        continuesUtterance
          ? "border-l-2 border-border ml-3"
          : "border-l-2 border-transparent mt-2"
      }${highlighted ? " bg-highlight" : ""}`}
    >
      <div className="flex items-center gap-2 text-xs text-fg-muted">
        {label && <span className="font-medium text-fg">{label}</span>}
        {line.edited && (
          <span className="rounded bg-warning-subtle px-1.5 py-0.5 text-warning">
            edited
          </span>
        )}
      </div>
      <p className="text-sm text-fg">{line.text}</p>
    </button>
  );
}
