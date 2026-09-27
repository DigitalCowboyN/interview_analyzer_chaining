import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

// ADR-0030 / spec §E: colors come from semantic tokens only, so both themes
// stay readable. Raw palette classes are how dark mode broke before.
const RAW_PALETTE =
  /\b(?:text|bg|border|divide|ring|from|to|via)-(?:neutral|gray|zinc|slate|stone|red|green|amber|yellow|blue|emerald|sky|indigo)-\d{2,3}\b|\b(?:text|bg)-(?:white|black)\b/g;

function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return name === "__tests__" ? [] : tsxFiles(full);
    return full.endsWith(".tsx") ? [full] : [];
  });
}

describe("palette guard", () => {
  it("no component uses raw Tailwind palette classes", () => {
    const srcDir = path.resolve(__dirname, "..");
    const offenders = tsxFiles(srcDir).flatMap((file) => {
      const hits = readFileSync(file, "utf8").match(RAW_PALETTE) ?? [];
      return hits.map((hit) => `${path.relative(srcDir, file)}: ${hit}`);
    });
    expect(offenders).toEqual([]);
  });
});
