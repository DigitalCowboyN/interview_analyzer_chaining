/** Lens node types in display order (spec §B). Types not listed here still
 * render — after these, labeled by their raw node_type. */
export const INSIGHT_TYPES = [
  { nodeType: "Decision", label: "Decisions", chip: "decisions" },
  { nodeType: "ActionItem", label: "Action items", chip: "actions" },
  { nodeType: "Objective", label: "Objectives", chip: "objectives" },
  { nodeType: "FollowUp", label: "Follow-ups", chip: "follow-ups" },
  { nodeType: "Goal", label: "Goals", chip: "goals" },
  { nodeType: "PainPoint", label: "Pain points", chip: "pain points" },
  { nodeType: "NotableQuote", label: "Quotes", chip: "quotes" },
] as const;

const KNOWN = new Map<string, { label: string; chip: string }>(
  INSIGHT_TYPES.map((t) => [t.nodeType, t]),
);

/** Node types present in `nodeTypes`, canonical ones first, then the rest alphabetically. */
export function orderNodeTypes(nodeTypes: Iterable<string>): string[] {
  const present = new Set(nodeTypes);
  const known = INSIGHT_TYPES.map((t) => t.nodeType as string).filter((t) => present.has(t));
  const unknown = [...present].filter((t) => !KNOWN.has(t)).sort();
  return [...known, ...unknown];
}

export function insightLabel(nodeType: string): string {
  return KNOWN.get(nodeType)?.label ?? nodeType;
}

export function insightChip(nodeType: string): string {
  return KNOWN.get(nodeType)?.chip ?? nodeType;
}
