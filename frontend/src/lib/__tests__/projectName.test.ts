import { describe, it, expect } from "vitest";
import { displayProjectName } from "@/lib/projectName";

describe("displayProjectName", () => {
  it("title-cases real project ids", () => {
    expect(displayProjectName({ project_id: "samples", kind: "real" })).toBe("Samples");
    expect(displayProjectName({ project_id: "real-interviews", kind: "real" })).toBe("Real Interviews");
    expect(displayProjectName({ project_id: "q3_research", kind: "real" })).toBe("Q3 Research");
  });

  it("labels test runs by suite and short id", () => {
    expect(
      displayProjectName({ project_id: "ui-smoke-150548b2-230b", kind: "test", suite: "ui-smoke" }),
    ).toBe("ui-smoke · 150548b2");
  });

  it("names the bucket", () => {
    expect(displayProjectName({ project_id: "test-runs" })).toBe("Test runs");
  });

  it("falls back to the raw id when kind is unknown", () => {
    expect(displayProjectName({ project_id: "whatever-1" })).toBe("whatever-1");
  });
});
