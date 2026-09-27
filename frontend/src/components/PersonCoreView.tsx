import Link from "next/link";
import { routes } from "@/lib/routes";
import type { PersonLink } from "@/hooks/usePersonDetail";

/**
 * Person CORE view (not a card): identity facts (linked speakers per
 * interview) plus a loose link to the persona profile. This is a link
 * BETWEEN views (navigates to the persona core view) — never an embed of
 * one entity's display inside the other's (domain-model rule; the m:n
 * future means a person may one day link to several persona profiles).
 */
export function PersonCoreView({
  projectId,
  personId,
  links,
  contributesToPersona,
}: {
  projectId: string;
  personId: string;
  links: PersonLink[];
  contributesToPersona: boolean;
}) {
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-sm font-semibold uppercase text-fg-muted">
          Linked speakers ({links.length})
        </h2>
        {links.length === 0 ? (
          <p className="mt-2 text-sm text-fg-muted">No linked speakers.</p>
        ) : (
          <ul className="mt-2 space-y-2">
            {links.map((link) => (
              <li
                key={`${link.interview_id}-${link.speaker_id}`}
                className="flex flex-wrap items-center justify-between gap-2 rounded border border-border p-3 bg-surface"
              >
                <span className="text-sm text-fg">
                  {link.speaker_display_name}
                </span>
                <span className="rounded bg-accent-subtle px-1.5 py-0.5 text-xs text-accent">
                  {link.interview_title}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="text-sm font-semibold uppercase text-fg-muted">Persona profile</h2>
        {contributesToPersona ? (
          <Link
            href={routes.persona(projectId, personId)}
            className="mt-2 inline-block text-sm text-accent hover:underline"
          >
            View persona profile →
          </Link>
        ) : (
          <p className="mt-2 text-sm text-fg-muted">
            No persona profile yet for this person.
          </p>
        )}
      </section>
    </div>
  );
}
