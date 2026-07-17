/**
 * Central query-key builders. TanStack Query keys are just arrays, but
 * centralizing them here keeps invalidation targets consistent across
 * hooks (e.g. the Task 5 intent pattern invalidates a `/ui` read after a
 * correction settles — it needs the exact same key the fetching hook used).
 */
export const queryKeys = {
  projects: () => ["projects"] as const,
  interviews: (projectId: string) => ["projects", projectId, "interviews"] as const,
  transcript: (interviewId: string) => ["interviews", interviewId, "transcript"] as const,
  personas: (projectId: string) => ["projects", projectId, "personas"] as const,
  persona: (projectId: string, personId: string) =>
    ["personas", projectId, personId] as const,
  persons: (projectId: string) => ["projects", projectId, "persons"] as const,
  person: (projectId: string, personId: string) =>
    ["persons", projectId, personId] as const,
};
