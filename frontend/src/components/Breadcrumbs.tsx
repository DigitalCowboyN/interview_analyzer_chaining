import Link from "next/link";

export interface BreadcrumbItem {
  label: string;
  /** Omit for the current (non-linked) trailing crumb. */
  href?: string;
}

/** Shared breadcrumb trail: `Workbench / {project} / {interview}`. */
export function Breadcrumbs({ items }: { items: BreadcrumbItem[] }) {
  return (
    <nav aria-label="Breadcrumb" className="mb-4 text-sm text-neutral-500">
      <ol className="flex flex-wrap items-center gap-1">
        {items.map((item, index) => (
          <li key={`${item.label}-${index}`} className="flex items-center gap-1">
            {index > 0 && <span aria-hidden="true">/</span>}
            {item.href ? (
              <Link href={item.href} className="hover:text-neutral-900">
                {item.label}
              </Link>
            ) : (
              <span className="font-medium text-neutral-900">{item.label}</span>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
