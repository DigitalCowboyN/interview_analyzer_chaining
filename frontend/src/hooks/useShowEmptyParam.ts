"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";

/**
 * Holds a page's "show empty rows" toggle in `?empty=1` (same pattern as
 * `setParam` in the transcript page: `router.replace` so Back doesn't pile
 * up history entries, and other params are preserved).
 */
export function useShowEmptyParam() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const showEmpty = searchParams.get("empty") === "1";

  function toggleShowEmpty() {
    const next = new URLSearchParams(searchParams.toString());
    if (showEmpty) next.delete("empty");
    else next.set("empty", "1");
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }

  return { showEmpty, toggleShowEmpty };
}
