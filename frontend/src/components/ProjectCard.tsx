import Link from "next/link";

/** Landing card for one project (or the Test runs bucket). */
export function ProjectCard({
  href,
  name,
  interviewCount,
  subtle = false,
}: {
  href: string;
  name: string;
  interviewCount: number;
  subtle?: boolean;
}) {
  return (
    <Link
      href={href}
      className={`block rounded-lg border border-border p-4 hover:border-accent ${
        subtle ? "bg-surface-raised" : "bg-surface"
      }`}
    >
      <span className="block font-medium text-fg">{name}</span>
      <span className="mt-1 block text-sm text-fg-muted">
        {interviewCount} {interviewCount === 1 ? "interview" : "interviews"}
      </span>
    </Link>
  );
}
