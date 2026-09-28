import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InsightsPanel } from "@/components/InsightsPanel";
import type { Insight } from "@/lib/insights";

const items: Insight[] = [
  { item_id: "d1", node_type: "Decision", lens: "meeting_minutes", text: "Ship CSV export", confidence: 0.92, locked: true, supporting_fragment_ids: ["f1"] },
  { item_id: "a1", node_type: "ActionItem", lens: "meeting_minutes", text: "Ravi drafts spec", confidence: 0.8, locked: false, supporting_fragment_ids: ["f2"] },
];

describe("InsightsPanel", () => {
  it("renders groups with counts and items", () => {
    render(<InsightsPanel insights={items} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByRole("heading", { name: "Decisions (1)" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Action items (1)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ship CSV export/ })).toHaveTextContent("locked");
  });

  it("calls onSelect and marks the selected item", async () => {
    const onSelect = vi.fn();
    const { rerender } = render(<InsightsPanel insights={items} selectedId={null} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /Ravi drafts spec/ }));
    expect(onSelect).toHaveBeenCalledWith(items[1]);
    rerender(<InsightsPanel insights={items} selectedId="a1" onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: /Ravi drafts spec/ })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows a quiet empty state", () => {
    render(<InsightsPanel insights={[]} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByText("No insights yet.")).toBeInTheDocument();
  });

  it("shows a muted loading state instead of the empty state while loading", () => {
    render(<InsightsPanel insights={[]} selectedId={null} onSelect={() => {}} isLoading />);
    expect(screen.getByText("Loading insights…")).toBeInTheDocument();
    expect(screen.queryByText("No insights yet.")).not.toBeInTheDocument();
  });
});
