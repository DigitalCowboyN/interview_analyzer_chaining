import Link from "next/link";
import { routes } from "@/lib/routes";
import type { PersonaCardData } from "@/hooks/usePersonaCards";

/**
 * Persona cards: a discovery grid, never the core display (domain-model
 * rule — see docs/superpowers/plans/2026-07-17-m50-ui-scaffolding.md Global
 * Constraints). Each card links through to the persona core view.
 */
export function PersonaCardGrid({
  projectId,
  personas,
}: {
  projectId: string;
  personas: PersonaCardData[];
}) {
  return (
    <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {personas.map((persona) => (
        <li key={persona.person_id}>
          <Link
            href={routes.persona(projectId, persona.person_id)}
            className="block rounded border border-border p-4 hover:bg-surface-raised"
          >
            <h3 className="font-medium text-fg">{persona.display_name}</h3>
            <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
              <div className="flex items-center gap-1">
                <dt>Traits</dt>
                <dd className="font-medium text-fg">{persona.trait_count}</dd>
              </div>
              <div className="flex items-center gap-1">
                <dt>Goals</dt>
                <dd className="font-medium text-fg">{persona.goal_count}</dd>
              </div>
              <div className="flex items-center gap-1">
                <dt>Pain points</dt>
                <dd className="font-medium text-fg">{persona.pain_point_count}</dd>
              </div>
              <div className="flex items-center gap-1">
                <dt>Quotes</dt>
                <dd className="font-medium text-fg">{persona.quote_count}</dd>
              </div>
            </dl>
            {persona.representative_quote && (
              <p className="mt-3 line-clamp-2 text-sm italic text-fg-muted">
                &ldquo;{persona.representative_quote}&rdquo;
              </p>
            )}
          </Link>
        </li>
      ))}
    </ul>
  );
}
