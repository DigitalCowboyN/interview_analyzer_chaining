"use client";

import Link from "next/link";
import { IdentitySwitcher } from "@/identity/IdentitySwitcher";
import { routes } from "@/lib/routes";

// Interim: Workbench/Gallery nav is gone now that routes are project-scoped
// (ADR-0030); a later task rewrites this header with a project switcher.
export function AppHeader() {
  return (
    <header className="flex items-center justify-between border-b border-border px-6 py-3 bg-surface">
      <div className="flex items-center gap-8">
        <Link href={routes.home()} className="font-semibold text-fg">
          Interview Analyzer
        </Link>
      </div>
      <IdentitySwitcher />
    </header>
  );
}
