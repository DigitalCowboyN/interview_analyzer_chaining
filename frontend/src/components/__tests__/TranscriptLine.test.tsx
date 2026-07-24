import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TranscriptLine } from "@/components/TranscriptLine";
import type { TranscriptLineData } from "@/hooks/useTranscript";

function makeLine(overrides: Partial<TranscriptLineData> = {}): TranscriptLineData {
  return {
    fragment_id: "f1",
    sequence_order: 0,
    text: "Hello there",
    speaker: { speaker_id: "s1", display_name: "Speaker A" },
    person: null,
    utterance_id: null,
    segment: null,
    entities: [],
    lens_items: [],
    edited: false,
    ...overrides,
  };
}

describe("TranscriptLine", () => {
  it("renders speaker label without a person suffix when unlinked", () => {
    render(
      <TranscriptLine line={makeLine()} continuesUtterance={false} onSelect={vi.fn()} />,
    );
    expect(screen.getByText("Speaker A")).toBeInTheDocument();
    expect(screen.getByText("Hello there")).toBeInTheDocument();
  });

  it('renders "Speaker (Person)" when a person is linked', () => {
    render(
      <TranscriptLine
        line={makeLine({
          person: { person_id: "p1", display_name: "Jane Doe" },
        })}
        continuesUtterance={false}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByText("Speaker A (Jane Doe)")).toBeInTheDocument();
  });

  it("shows an edited badge only when edited is true", () => {
    const { rerender } = render(
      <TranscriptLine line={makeLine({ edited: true })} continuesUtterance={false} onSelect={vi.fn()} />,
    );
    expect(screen.getByText("edited")).toBeInTheDocument();

    rerender(
      <TranscriptLine line={makeLine({ edited: false })} continuesUtterance={false} onSelect={vi.fn()} />,
    );
    expect(screen.queryByText("edited")).not.toBeInTheDocument();
  });

  it("indicates utterance grouping via data attributes when continuing an utterance", () => {
    render(
      <TranscriptLine
        line={makeLine({ utterance_id: "u1" })}
        continuesUtterance={true}
        onSelect={vi.fn()}
      />,
    );
    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("data-utterance-id", "u1");
    expect(button).toHaveAttribute("data-continues-utterance", "true");
  });

  it("calls onSelect with the line when clicked", async () => {
    const onSelect = vi.fn();
    const line = makeLine();
    render(<TranscriptLine line={line} continuesUtterance={false} onSelect={onSelect} />);

    await userEvent.click(screen.getByRole("button"));
    expect(onSelect).toHaveBeenCalledWith(line);
  });
});
