"use client";

import Link from "next/link";
import { IdentitySwitcher } from "@/identity/IdentitySwitcher";
import { ProjectSwitcher } from "@/components/ProjectSwitcher";
import { routes } from "@/lib/routes";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-surface px-6 py-3">
      <div className="flex items-center gap-6">
        <Link href={routes.home()} className="font-semibold text-fg">
          Interview Analyzer
        </Link>
        <ProjectSwitcher />
      </div>
      <IdentitySwitcher />
    </header>
  );
}
