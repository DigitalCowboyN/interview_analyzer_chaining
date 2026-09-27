import { describe, it, expect } from "vitest";
import { routes, TEST_RUNS_ID } from "@/lib/routes";

describe("routes", () => {
  it("builds project-scoped paths", () => {
    expect(routes.home()).toBe("/");
    expect(routes.project("samples")).toBe("/projects/samples");
    expect(routes.interview("samples", "i1")).toBe("/projects/samples/interviews/i1");
    expect(routes.personas("p")).toBe("/projects/p/personas");
    expect(routes.persona("p", "x")).toBe("/projects/p/personas/x");
    expect(routes.people("p")).toBe("/projects/p/people");
    expect(routes.person("p", "x")).toBe("/projects/p/people/x");
    expect(routes.review("p")).toBe("/projects/p/review");
    expect(routes.testRuns()).toBe(`/projects/${TEST_RUNS_ID}`);
  });

  it("encodes URL-meaningful characters in ids", () => {
    expect(routes.interview("a/b c", "i%1")).toBe("/projects/a%2Fb%20c/interviews/i%251");
  });
});
