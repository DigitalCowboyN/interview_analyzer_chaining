import { describe, it, expect } from "vitest";
import { groupInsights, type Insight } from "@/lib/insights";

const mk = (item_id: string, node_type: string): Insight => ({
  item_id, node_type, lens: "persona", text: item_id, confidence: 0.9, locked: false,
  supporting_fragment_ids: [],
});

describe("groupInsights", () => {
  it("groups in canonical order and keeps unknown types in a trailing group", () => {
    const groups = groupInsights([mk("q", "NotableQuote"), mk("t", "Trait"), mk("d", "Decision"), mk("d2", "Decision")]);
    expect(groups.map((g) => [g.label, g.items.length])).toEqual([
      ["Decisions", 2], ["Quotes", 1], ["Trait", 1],
    ]);
  });

  it("returns no groups for no items", () => {
    expect(groupInsights([])).toEqual([]);
  });
});
