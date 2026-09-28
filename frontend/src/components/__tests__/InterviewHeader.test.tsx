import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterviewHeader } from "@/components/InterviewHeader";

describe("InterviewHeader", () => {
  it("shows title, participants, date, and line count; hides empty metadata", () => {
    render(<InterviewHeader title="Sync" participants={["Maya", "Ravi"]} date="Sep 27, 2026" lineCount={81} metadata={{}} />);
    expect(screen.getByRole("heading", { level: 1, name: "Sync" })).toBeInTheDocument();
    expect(screen.getByText("Maya, Ravi · Sep 27, 2026 · 81 lines")).toBeInTheDocument();
    expect(screen.queryByText(/No metadata/)).not.toBeInTheDocument();
  });

  it("lists metadata when present", () => {
    render(<InterviewHeader title="T" participants={[]} date="" lineCount={1} metadata={{ source: "zoom" }} />);
    expect(screen.getByText("source")).toBeInTheDocument();
    expect(screen.getByText("zoom")).toBeInTheDocument();
  });
});
