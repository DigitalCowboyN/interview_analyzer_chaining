/** Topic heading rendered above the first line of a new segment. */
export function SegmentHeading({ topic }: { topic: string | null }) {
  return (
    <h2 className="mt-6 mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-500">
      {topic ?? "Untitled segment"}
    </h2>
  );
}
