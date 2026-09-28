import { describe, it, expect } from "vitest";
import { formatDate } from "@/lib/formatDate";

describe("formatDate", () => {
  it("formats ISO with microseconds (Neo4j toString) as a short date", () => {
    expect(formatDate("2026-09-27T10:12:13.623624Z")).toBe("Sep 27, 2026");
  });
  it("formats plain dates", () => {
    expect(formatDate("2026-01-05")).toBe("Jan 5, 2026");
  });
  it("returns the raw string when unparseable and empty for nullish", () => {
    expect(formatDate("not a date")).toBe("not a date");
    expect(formatDate(null)).toBe("");
    expect(formatDate(undefined)).toBe("");
  });
});
