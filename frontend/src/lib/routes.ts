/** Every in-app URL (ADR-0034: the project is the top-level nav scope).
 * Components link via these builders only — never hand-built strings. */
// governed-by: ADR-0034
const e = encodeURIComponent;

export const TEST_RUNS_ID = "test-runs";

export const routes = {
  home: () => "/",
  project: (projectId: string) => `/projects/${e(projectId)}`,
  interview: (projectId: string, interviewId: string) =>
    `/projects/${e(projectId)}/interviews/${e(interviewId)}`,
  personas: (projectId: string) => `/projects/${e(projectId)}/personas`,
  persona: (projectId: string, personId: string) =>
    `/projects/${e(projectId)}/personas/${e(personId)}`,
  people: (projectId: string) => `/projects/${e(projectId)}/people`,
  person: (projectId: string, personId: string) =>
    `/projects/${e(projectId)}/people/${e(personId)}`,
  review: (projectId: string) => `/projects/${e(projectId)}/review`,
  testRuns: () => `/projects/${TEST_RUNS_ID}`,
};
