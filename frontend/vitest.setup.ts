import "@testing-library/jest-dom/vitest";

/**
 * jsdom does not implement `EventSource` (used by useLiveInvalidation,
 * M5.1 Task 5). Any test that mounts a page rendering the two workbench
 * screens now indirectly constructs one; without a global stub, `new
 * EventSource(...)` throws `ReferenceError: EventSource is not defined`.
 * This is a harmless no-op stub — it never calls onopen/onmessage/onerror
 * on its own — so page tests that don't care about live-invalidation
 * behavior stay green untouched. Tests that DO care about that behavior
 * (useLiveInvalidation.test.tsx) install their own richer mock via
 * `vi.stubGlobal`, which shadows this one for the duration of that file.
 */
class NoopEventSource {
  constructor(public url: string) {}
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  close(): void {}
}

// @ts-expect-error -- test-only stub, not a spec-complete EventSource
globalThis.EventSource = NoopEventSource;
