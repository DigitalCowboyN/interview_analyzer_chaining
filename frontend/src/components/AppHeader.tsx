"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { IdentitySwitcher } from "@/identity/IdentitySwitcher";

const NAV_LINKS = [
  { href: "/workbench", label: "Workbench" },
  { href: "/gallery", label: "Gallery" },
] as const;

export function AppHeader() {
  const pathname = usePathname();

  return (
    <header className="flex items-center justify-between border-b border-neutral-200 px-6 py-3">
      <div className="flex items-center gap-8">
        <span className="font-semibold">Interview Analyzer</span>
        <nav className="flex items-center gap-4 text-sm">
          {NAV_LINKS.map((link) => {
            const isActive = pathname?.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={
                  isActive
                    ? "font-medium text-neutral-900"
                    : "text-neutral-500 hover:text-neutral-900"
                }
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
      <IdentitySwitcher />
    </header>
  );
}
