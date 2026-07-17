"use client";

import { useParams } from "next/navigation";
import { useState } from "react";
import { useTranscript } from "@/hooks/useTranscript";
import type { TranscriptLineData } from "@/hooks/useTranscript";
import { StateGate } from "@/components/StateGate";
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { MetadataPanel } from "@/components/MetadataPanel";
import { SegmentHeading } from "@/components/SegmentHeading";
import { TranscriptLine } from "@/components/TranscriptLine";
import { LineDetailPanel } from "@/components/LineDetailPanel";

/** Transcript screen: the workbench's core, read-only display of an interview. */
export default function TranscriptPage() {
  const { projectId, interviewId } = useParams<{
    projectId: string;
    interviewId: string;
  }>();
  const { data: transcript, isLoading, isError, error } = useTranscript(interviewId);
  const [selectedLine, setSelectedLine] = useState<TranscriptLineData | null>(null);

  return (
    <div className="flex">
      <div className="flex-1 p-6">
        <Breadcrumbs
          items={[
            { label: "Workbench", href: "/workbench" },
            { label: projectId, href: `/workbench/${encodeURIComponent(projectId)}` },
            { label: transcript?.title ?? interviewId },
          ]}
        />
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={transcript?.lines.length === 0}
          emptyFallback={
            <div className="p-4 text-sm text-neutral-500">
              This interview has no transcript lines yet.
            </div>
          }
        >
          {transcript && (
            <>
              <MetadataPanel title={transcript.title} metadata={transcript.metadata} />
              <div className="mt-4">
                {transcript.lines.map((line, index) => {
                  const previous = transcript.lines[index - 1];
                  // Segment heading precedes the first line of each segment —
                  // derived from the segment field changing between
                  // consecutive lines (a null->non-null or id change).
                  const startsNewSegment =
                    line.segment !== null &&
                    (!previous || previous.segment?.segment_id !== line.segment.segment_id);
                  const continuesUtterance = Boolean(
                    line.utterance_id &&
                      previous &&
                      previous.utterance_id === line.utterance_id,
                  );

                  return (
                    <div key={line.fragment_id}>
                      {startsNewSegment && (
                        <SegmentHeading topic={line.segment?.topic ?? null} />
                      )}
                      <TranscriptLine
                        line={line}
                        continuesUtterance={continuesUtterance}
                        onSelect={setSelectedLine}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </StateGate>
      </div>
      {selectedLine && (
        <LineDetailPanel
          interviewId={interviewId}
          line={selectedLine}
          onClose={() => setSelectedLine(null)}
        />
      )}
    </div>
  );
}
