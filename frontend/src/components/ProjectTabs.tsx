"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { routes } from "@/lib/routes";

/** Per-project tabs; each tab is a URL so Back/Forward walk tab history. */
export function ProjectTabs({ projectId }: { projectId: string }) {
  const pathname = usePathname() ?? "";
  const tabs = [
    { label: "Interviews", href: routes.project(projectId), match: (p: string) =>
        p === routes.project(projectId) || p.startsWith(`${routes.project(projectId)}/interviews`) },
    { label: "Personas", href: routes.personas(projectId), match: (p: string) => p.startsWith(routes.personas(projectId)) },
    { label: "People", href: routes.people(projectId), match: (p: string) => p.startsWith(routes.people(projectId)) },
    { label: "Review", href: routes.review(projectId), match: (p: string) => p.startsWith(routes.review(projectId)) },
  ];

  return (
    <nav aria-label="Project sections" className="flex gap-1 border-b border-border px-6">
      {tabs.map((tab) => {
        const active = tab.match(pathname);
        return (
          <Link
            key={tab.label}
            href={tab.href}
            aria-current={active ? "page" : undefined}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              active
                ? "border-accent font-medium text-fg"
                : "border-transparent text-fg-muted hover:text-fg"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
