import { describe, it, expect, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useShowEmptyParam } from "@/hooks/useShowEmptyParam";

const replace = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(),
  useRouter: vi.fn(),
  useSearchParams: vi.fn(),
}));

function mockNav(search: string) {
  vi.mocked(usePathname).mockReturnValue("/projects/p1");
  vi.mocked(useRouter).mockReturnValue({ replace } as never);
  vi.mocked(useSearchParams).mockReturnValue(new URLSearchParams(search) as never);
}

describe("useShowEmptyParam", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    replace.mockClear();
  });

  it("showEmpty is false when ?empty= is absent", () => {
    mockNav("");
    const { result } = renderHook(() => useShowEmptyParam());
    expect(result.current.showEmpty).toBe(false);
  });

  it("showEmpty is true when ?empty=1", () => {
    mockNav("empty=1");
    const { result } = renderHook(() => useShowEmptyParam());
    expect(result.current.showEmpty).toBe(true);
  });

  it("toggling on replaces the URL with ?empty=1, preserving other params", () => {
    mockNav("sort=recent");
    const { result } = renderHook(() => useShowEmptyParam());
    act(() => {
      result.current.toggleShowEmpty();
    });
    expect(replace).toHaveBeenCalledWith("/projects/p1?sort=recent&empty=1", { scroll: false });
  });

  it("toggling off removes ?empty from the URL", () => {
    mockNav("empty=1&sort=recent");
    const { result } = renderHook(() => useShowEmptyParam());
    act(() => {
      result.current.toggleShowEmpty();
    });
    expect(replace).toHaveBeenCalledWith("/projects/p1?sort=recent", { scroll: false });
  });
});
